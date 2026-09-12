"""Xử lý, làm sạch và chia bộ dữ liệu lá lúa (Data Preparation Pipeline).

Quy trình kỹ thuật:
1. Giải nén an toàn file ZIP và xác định cấu trúc dataset nguồn.
2. Kiểm tra tính hợp lệ của annotation:
   - valid: có ít nhất 1 bbox thuộc 2 lớp bệnh mục tiêu.
   - negative: ảnh không có bệnh mục tiêu.
   - invalid: nhãn lỗi/thiếu -> loại bỏ để tránh biến ảnh bệnh thành background.
3. Loại bỏ ảnh trùng lặp tuyệt đối (SHA-256) và nhóm ảnh biến thể (original_key + pHash).
4. Phân chia Train/Val/Test theo nhóm (Group-aware Split) nhằm chống rò rỉ dữ liệu (Data Leakage).
5. Xuất dataset chuẩn YOLOv8, manifest.csv, data.yaml và data_report.json.
"""

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable


from .annotations import build_class_map, parse_label_file_detailed
from .constants import (
    CLASS_NAMES,
    DEFAULT_ARCHIVES,
    IMAGE_EXTENSIONS,
    SEED,
    SPLIT_RATIOS,
    SPLITS,
    STATUS_INVALID,
    STATUS_NEGATIVE,
    STATUS_VALID,
)
from .deduplication import deduplicate_and_group
from .utils import (
    configure_utf8_console,
    perceptual_hash,
    safe_extract_zip,
    seed_everything,
    sha256_file,
    write_json,
)

Record = dict[str, Any]
DataReport = dict[str, Any]


def locate_dataset_root(directory: Path) -> Path:
    candidates = []
    for yaml_path in directory.rglob("data.yaml"):
        root = yaml_path.parent
        has_train = (root / "train" / "images").exists()
        has_validation = (root / "valid" / "images").exists() or (root / "val" / "images").exists()
        has_test = (root / "test" / "images").exists()
        if has_train and has_validation and has_test:
            candidates.append(root)
    candidates = sorted(set(candidates), key=lambda path: len(path.parts))
    if len(candidates) != 1:
        raise RuntimeError(f"Cần đúng một bộ dữ liệu YOLO trong {directory}, tìm thấy {candidates}")
    return candidates[0]


def prepare_sources(archives: list[Path], extract_root: Path) -> list[dict[str, Any]]:
    if not archives:
        raise ValueError("Cần cung cấp ít nhất một file ZIP")
    sources = []
    used_names: set[str] = set()
    for archive in archives:
        name = re.sub(r"[^A-Za-z0-9_-]+", "_", archive.stem).lower()
        if name in used_names:
            raise ValueError(f"Hai file ZIP tạo cùng tên nguồn: {name}")
        used_names.add(name)
        destination = extract_root / name
        temporary = extract_root / f"{name}__extracting"
        if temporary.exists():
            shutil.rmtree(temporary)
        print(f"Giải nén {archive}...")
        safe_extract_zip(archive, temporary)
        locate_dataset_root(temporary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.rename(destination)
        sources.append(
            {
                "name": name,
                "root": locate_dataset_root(destination),
                "archive": str(archive.resolve()),
                "archive_sha256": sha256_file(archive),
            }
        )
    return sources


def collect_records(
    sources: list[dict[str, Any]],
    keep_negatives: bool = True,
) -> tuple[list[Record], DataReport]:
    records: list[Record] = []
    report: DataReport = {
        "total_source_images_scanned": 0,
        "valid_images": 0,
        "negative_images": 0,
        "invalid_or_missing_annotations": 0,
        "missing_label_files": [],
        "invalid_labels": [],
        "corrupt_images": [],
        "clipped_boxes": 0,
        "bbox_count": 0,
        "polygon_count": 0,
        "duplicate_annotation_lines_removed": 0,
        "quarantined_records": [],
    }

    for source in sources:
        _, class_map = build_class_map(source["root"])
        for old_split in ("train", "valid", "val", "test"):
            image_dir = source["root"] / old_split / "images"
            label_dir = source["root"] / old_split / "labels"
            if not image_dir.exists():
                continue
            images = sorted(
                path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for image_path in tqdm(images, desc=f"Đọc {source['name']}/{old_split}"):
                report["total_source_images_scanned"] += 1
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    report["missing_label_files"].append(str(label_path))

                annotations, errors, duplicate_count, unknown_count = parse_label_file_detailed(
                    label_path, class_map
                )
                report["invalid_labels"].extend(errors)
                report["duplicate_annotation_lines_removed"] += duplicate_count
                report["clipped_boxes"] += sum(
                    str(a["source_type"]).endswith("_clipped") for a in annotations
                )
                for a in annotations:
                    if "polygon" in str(a.get("source_type", "")):
                        report["polygon_count"] += 1
                    else:
                        report["bbox_count"] += 1

                # Phân loại annotation status đơn giản: valid / negative / invalid
                if not label_path.exists() or errors:
                    annotation_status = STATUS_INVALID
                elif annotations:
                    annotation_status = STATUS_VALID
                else:
                    # File nhãn rỗng tồn tại hợp lệ là negative thực sự
                    annotation_status = STATUS_NEGATIVE

                if annotation_status == STATUS_INVALID:
                    report["invalid_or_missing_annotations"] += 1
                    report["quarantined_records"].append(
                        {
                            "image_path": str(image_path),
                            "label_path": str(label_path),
                            "errors": errors,
                        }
                    )
                    # Không để nhãn lỗi hay file thiếu trở thành negative âm thầm
                    continue

                if annotation_status == STATUS_NEGATIVE and not keep_negatives:
                    continue

                if annotation_status == STATUS_VALID:
                    report["valid_images"] += 1
                else:
                    report["negative_images"] += 1

                try:
                    with Image.open(image_path) as img:
                        rgb = img.convert("RGB")
                        width, height = rgb.size
                        phash = perceptual_hash(rgb)
                except Exception as exc:
                    report["corrupt_images"].append({"path": str(image_path), "error": str(exc)})
                    continue

                records.append(
                    {
                        "image_id": f"{source['name']}__{image_path.stem}",
                        "source_image_id": image_path.stem,
                        "source": source["name"],
                        "old_split": old_split,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "width": width,
                        "height": height,
                        "sha256": sha256_file(image_path),
                        "phash": phash,
                        "original_key": f"{source['name']}:{image_path.stem.split('.rf.')[0]}",
                        "annotation_status": annotation_status,
                        "is_negative": annotation_status == STATUS_NEGATIVE,
                        "annotations": annotations,
                    }
                )

    if not records:
        raise RuntimeError("Không tìm thấy ảnh hợp lệ trong các bộ dữ liệu nguồn")
    return records, report


def _group_summary(group: list[Record]) -> dict[str, Any]:
    source_counts = Counter(record["source"] for record in group)
    status_counts = Counter(record["annotation_status"] for record in group)
    class_counts = Counter(
        int(ann["class_id"]) for record in group for ann in record.get("annotations", [])
    )
    return {
        "images": len(group),
        "sources": source_counts,
        "statuses": status_counts,
        "classes": class_counts,
    }


def _split_loss(
    current: dict[str, float],
    addition: dict[str, float],
    target: dict[str, float],
) -> float:
    loss = 0.0
    for key, expected in target.items():
        if expected <= 0:
            continue
        observed = current.get(key, 0.0) + addition.get(key, 0.0)
        loss += ((observed - expected) / expected) ** 2
    return loss


def assign_splits(records: list[Record], seed: int = SEED) -> None:
    """Chia Group + Source + Class aware bằng greedy assignment tất định."""
    if not records:
        raise ValueError("Không có record để chia dữ liệu")

    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)
    if len(grouped) < 3:
        raise ValueError("Cần ít nhất 3 group độc lập để tạo train/val/test")

    summaries = {group_id: _group_summary(items) for group_id, items in grouped.items()}
    sources = sorted({record["source"] for record in records})
    dimensions = ["images"]
    dimensions.extend(f"class_{class_id}" for class_id in range(len(CLASS_NAMES)))
    dimensions.extend(f"source_{source}" for source in sources)

    total: dict[str, float] = {dim: 0.0 for dim in dimensions}
    vectors: dict[str, dict[str, float]] = {}
    for group_id, summary in summaries.items():
        vec = {dim: 0.0 for dim in dimensions}
        vec["images"] = float(summary["images"])
        for class_id, count in summary["classes"].items():
            vec[f"class_{class_id}"] = float(count)
        for src, count in summary["sources"].items():
            vec[f"source_{src}"] = float(count)
        vectors[group_id] = vec
        for dim, val in vec.items():
            total[dim] += val

    target_by_split = {
        split: {dim: total[dim] * SPLIT_RATIOS[split] for dim in dimensions} for split in SPLITS
    }
    assigned: dict[str, str] = {}
    current = {split: {dim: 0.0 for dim in dimensions} for split in SPLITS}
    group_counts = {split: 0 for split in SPLITS}

    ordered_groups = sorted(
        grouped,
        key=lambda gid: (
            -vectors[gid]["images"],
            -sum(vectors[gid][k] for k in dimensions if k.startswith("class_")),
            gid,
        ),
    )
    rotation = seed % len(ordered_groups)
    ordered_groups = ordered_groups[rotation:] + ordered_groups[:rotation]

    for pos, group_id in enumerate(ordered_groups):
        remaining = len(ordered_groups) - pos
        empty_splits = [s for s in SPLITS if group_counts[s] == 0]
        candidate_splits = (
            empty_splits if remaining == len(empty_splits) and empty_splits else list(SPLITS)
        )
        chosen = min(
            candidate_splits,
            key=lambda s: (
                _split_loss(current[s], vectors[group_id], target_by_split[s]),
                group_counts[s],
                s,
            ),
        )
        assigned[group_id] = chosen
        group_counts[chosen] += 1
        for dim, val in vectors[group_id].items():
            current[chosen][dim] += val

    for record in records:
        record["split"] = assigned[record["group_id"]]


def _safe_name(record: Record) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(record["image_path"]).stem)[:90]
    ext = Path(record["image_path"]).suffix.lower()
    return f"{record['source']}__{stem}__{record['sha256'][:10]}{ext}"


def write_dataset(
    records: list[Record],
    report: DataReport,
    output: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} đã tồn tại. Dùng --overwrite nếu muốn tạo lại.")
    temporary = output.parent / f"{output.name}__building"
    if temporary.exists():
        shutil.rmtree(temporary)
    for split in SPLITS:
        (temporary / split / "images").mkdir(parents=True)
        (temporary / split / "labels").mkdir(parents=True)

    rows = []
    for record in tqdm(records, desc="Ghi dataset sạch"):
        name, split = _safe_name(record), record["split"]
        relative_image = Path(split) / "images" / name
        shutil.copy2(record["image_path"], temporary / relative_image)
        label_file = temporary / split / "labels" / f"{Path(name).stem}.txt"
        label_lines = (
            f"{ann['class_id']} {ann['x']:.6f} {ann['y']:.6f} {ann['w']:.6f} {ann['h']:.6f}\n"
            for ann in record.get("annotations", [])
        )
        label_file.write_text("".join(label_lines), encoding="utf-8")
        counts = Counter(a["class_id"] for a in record.get("annotations", []))
        rows.append(
            {
                "image_id": record["image_id"],
                "source_image_id": record["source_image_id"],
                "split": split,
                "source": record["source"],
                "old_split": record["old_split"],
                "group_id": record["group_id"],
                "original_key": record.get("original_key", ""),
                "output_image": relative_image.as_posix(),
                "source_image": record["image_path"],
                "sha256": record["sha256"],
                "phash": record["phash"],
                "width": record["width"],
                "height": record["height"],
                "annotation_status": record["annotation_status"],
                "is_negative": record["is_negative"],
                "instances_class_0": counts[0],
                "instances_class_1": counts[1],
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(temporary / "manifest.csv", index=False)

    # Thống kê split cho report
    split_summary = {}
    for split in SPLITS:
        subset = frame[frame["split"] == split]
        split_summary[split] = {
            "images": len(subset),
            "negative_images": int(subset["is_negative"].sum()),
            "instances_bacterial_leaf_blight": int(subset["instances_class_0"].sum()),
            "instances_brown_spot": int(subset["instances_class_1"].sum()),
        }

    clean_report = {
        "dataset_name": "rice_leaf_disease_detection",
        "classes": CLASS_NAMES,
        "summary": {
            "total_clean_images": len(frame),
            "exact_duplicates_removed": report.get("exact_duplicates_removed", 0),
            "near_duplicate_links": report.get("near_duplicate_links", 0),
            "invalid_or_missing_annotations_discarded": report.get(
                "invalid_or_missing_annotations", 0
            ),
            "clipped_boxes": report.get("clipped_boxes", 0),
        },
        "splits": split_summary,
    }
    write_json(temporary / "data_report.json", clean_report)

    # data.yaml chuẩn cho YOLOv8
    yaml_config = {
        "path": output.resolve().as_posix(),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    (temporary / "data.yaml").write_text(
        yaml.safe_dump(yaml_config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    validate_dataset(frame, temporary)

    backup = output.parent / f"{output.name}__backup"
    if backup.exists():
        shutil.rmtree(backup)
    if output.exists():
        output.rename(backup)
    try:
        temporary.rename(output)
    except Exception:
        if backup.exists() and not output.exists():
            backup.rename(output)
        raise
    if backup.exists():
        shutil.rmtree(backup)
    return frame


MIN_GROUPS_PER_SPLIT = {"train": 10, "val": 5, "test": 5}
MIN_INSTANCES_PER_CLASS = {"val": 20, "test": 20}


def validate_split_sizes(manifest: pd.DataFrame) -> None:
    group_counts = manifest.groupby("split")["group_id"].nunique()
    for split, minimum in MIN_GROUPS_PER_SPLIT.items():
        count = group_counts.get(split, 0)
        if count < minimum:
            raise ValueError(f"Tập {split} chỉ có {count} nhóm ảnh, yêu cầu tối thiểu {minimum}")

    for split, minimum in MIN_INSTANCES_PER_CLASS.items():
        if split not in manifest["split"].values:
            continue
        for class_id in range(len(CLASS_NAMES)):
            col = f"instances_class_{class_id}"
            if col in manifest.columns:
                instance_count = manifest.loc[manifest["split"] == split, col].sum()
                if instance_count < minimum:
                    cname = CLASS_NAMES[class_id]
                    raise ValueError(
                        f"Tập {split} chỉ có {instance_count} instance lớp {class_id} ({cname}), "
                        f"yêu cầu tối thiểu {minimum}"
                    )


def validate_dataset(manifest: pd.DataFrame, output: Path) -> None:
    if manifest.empty:
        raise ValueError("Manifest không có dữ liệu")
    if "annotation_status" in manifest.columns:
        invalid = manifest[manifest["annotation_status"] == STATUS_INVALID]
        if not invalid.empty:
            raise ValueError("Manifest chứa annotation thiếu/lỗi; các record này phải bị loại bỏ")
    if manifest.groupby("group_id")["split"].nunique().max() != 1:
        raise ValueError("Có nhóm ảnh xuất hiện ở nhiều tập dữ liệu (rò rỉ nhóm)")
    if manifest.groupby("sha256")["split"].nunique().max() != 1:
        raise ValueError("Có ảnh trùng SHA-256 xuất hiện ở nhiều tập dữ liệu")
    if (
        "original_key" in manifest.columns
        and manifest.groupby("original_key")["split"].nunique().max() != 1
    ):
        raise ValueError(
            "Có ảnh cùng original_key xuất hiện ở nhiều tập dữ liệu (rò rỉ augmentation)"
        )

    validate_split_sizes(manifest)

    for split in SPLITS:
        images = {p.stem for p in (output / split / "images").iterdir() if p.is_file()}
        labels = {p.stem for p in (output / split / "labels").glob("*.txt")}
        if images != labels:
            raise ValueError(f"Ảnh và nhãn không khớp ở tập {split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chuẩn hóa và chia bộ dữ liệu YOLO lá lúa")
    parser.add_argument("--archives", nargs="+", type=Path, default=list(DEFAULT_ARCHIVES))
    parser.add_argument("--output", type=Path, default=Path("data/processed/rice_leaf_detection"))
    parser.add_argument("--extract-dir", type=Path, default=Path("data/extracted"))
    parser.add_argument("--phash-distance", type=int, default=2)
    parser.add_argument("--drop-negatives", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    seed_everything(SEED)
    if args.phash_distance < 0:
        raise ValueError("--phash-distance không được âm")
    sources = prepare_sources(args.archives, args.extract_dir)
    records, report = collect_records(sources, keep_negatives=not args.drop_negatives)
    records, dedup_report = deduplicate_and_group(records, args.phash_distance)
    report.update(dedup_report)

    conflicts = report.get("annotation_conflicts", [])
    if conflicts:
        conflict_dir = Path("reports/data_conflicts")
        conflict_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "sha256": c["sha256"],
                    "reason": c.get("reason", "conflict"),
                    "paths": "; ".join(c.get("paths", [])),
                }
                for c in conflicts
            ]
        ).to_csv(conflict_dir / "conflicts.csv", index=False)
        print(f"Đã cách ly {len(conflicts)} xung đột nhãn vào reports/data_conflicts/conflicts.csv")

    assign_splits(records)
    manifest = write_dataset(records, report, args.output, overwrite=args.overwrite)

    print(f"\nDataset đã tạo tại {args.output.resolve()}")
    print(f"- Tổng số ảnh sạch: {len(manifest)}")
    print(f"- Đã loại {report.get('exact_duplicates_removed', 0)} ảnh trùng tuyệt đối (SHA-256)")
    print(f"- Gom nhóm {report.get('near_duplicate_links', 0)} liên kết pHash gần trùng")
    print(f"- Loại bỏ {report.get('invalid_or_missing_annotations', 0)} nhãn lỗi hoặc thiếu")


if __name__ == "__main__":
    main()
