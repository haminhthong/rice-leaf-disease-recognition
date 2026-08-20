import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .annotations import build_class_map, parse_label_file
from .constants import CLASS_NAMES, DEFAULT_ARCHIVES, IMAGE_EXTENSIONS, SEED, SPLIT_RATIOS, SPLITS
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
AuditReport = dict[str, Any]


def locate_dataset_root(directory: Path) -> Path:
    candidates = []
    for yaml_path in directory.rglob("data.yaml"):
        root = yaml_path.parent
        has_train = (root / "train" / "images").exists()
        has_validation = (
            (root / "valid" / "images").exists()
            or (root / "val" / "images").exists()
        )
        has_test = (root / "test" / "images").exists()
        if has_train and has_validation and has_test:
            candidates.append(root)
    candidates = sorted(set(candidates), key=lambda path: len(path.parts))
    if len(candidates) != 1:
        raise RuntimeError(f"Cần đúng một dataset YOLO trong {directory}, tìm thấy {candidates}")
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
            }
        )
    return sources


def collect_records(
    sources: list[dict[str, Any]],
    keep_negatives: bool,
) -> tuple[list[Record], AuditReport]:
    records: list[Record] = []
    audit: AuditReport = {
        "invalid_labels": [],
        "corrupt_images": [],
        "missing_label_files": [],
        "duplicate_annotation_lines_removed": 0,
        "clipped_boxes": 0,
        "source_classes": {},
    }
    for source in sources:
        names, class_map = build_class_map(source["root"])
        audit["source_classes"][source["name"]] = {"names": names, "old_to_new": class_map}
        for old_split in ("train", "valid", "val", "test"):
            image_dir = source["root"] / old_split / "images"
            label_dir = source["root"] / old_split / "labels"
            if not image_dir.exists():
                continue
            images = sorted(path for path in image_dir.iterdir()
                            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
            for image_path in tqdm(images, desc=f"Đọc {source['name']}/{old_split}"):
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.exists():
                    audit["missing_label_files"].append(str(label_path))
                annotations, errors, duplicate_count = parse_label_file(label_path, class_map)
                audit["invalid_labels"].extend(errors)
                audit["duplicate_annotation_lines_removed"] += duplicate_count
                audit["clipped_boxes"] += sum(
                    str(annotation["source_type"]).endswith("_clipped")
                    for annotation in annotations
                )
                if not annotations and not keep_negatives:
                    continue
                try:
                    with Image.open(image_path) as image:
                        rgb = image.convert("RGB")
                        width, height = rgb.size
                        phash = perceptual_hash(rgb)
                except Exception as exc:
                    audit["corrupt_images"].append({"path": str(image_path), "error": str(exc)})
                    continue
                records.append(
                    {
                        "source": source["name"],
                        "old_split": old_split,
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "width": width,
                        "height": height,
                        "sha256": sha256_file(image_path),
                        "phash": phash,
                        "original_key": (
                            f"{source['name']}:{image_path.stem.split('.rf.')[0]}"
                        ),
                        "annotations": annotations,
                    }
                )
    if not records:
        raise RuntimeError("Không tìm thấy ảnh hợp lệ trong các dataset nguồn")
    return records, audit


def _presence(group: list[Record]) -> str:
    classes = sorted({ann["class_id"] for record in group for ann in record["annotations"]})
    return "negative" if not classes else "classes_" + "_".join(map(str, classes))


def _split(
    items: list[str],
    size: float,
    strata: list[str],
    seed: int,
) -> tuple[list[str], list[str]]:
    if len(items) < 2:
        raise ValueError("Không đủ nhóm ảnh để chia dữ liệu")
    counts = Counter(strata)
    adjusted = [value if counts[value] >= 2 else "rare" for value in strata]
    adjusted_counts = Counter(adjusted)
    stratify = adjusted if len(adjusted_counts) > 1 and min(adjusted_counts.values()) >= 2 else None
    return train_test_split(items, test_size=size, random_state=seed, stratify=stratify)


def assign_splits(records: list[Record]) -> None:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)
    group_ids = sorted(grouped)
    strata = [_presence(grouped[group]) for group in group_ids]
    holdout_ratio = SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"]
    train, holdout = _split(group_ids, holdout_ratio, strata, SEED)
    holdout_strata = [_presence(grouped[group]) for group in holdout]
    val, test = _split(holdout, SPLIT_RATIOS["test"] / holdout_ratio, holdout_strata, SEED + 1)
    assignments = {group: "train" for group in train}
    assignments.update({group: "val" for group in val})
    assignments.update({group: "test" for group in test})
    for record in records:
        record["split"] = assignments[record["group_id"]]


def _safe_name(record: Record) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(record["image_path"]).stem)[:90]
    return f"{record['source']}__{stem}__{record['sha256'][:10]}{Path(record['image_path']).suffix.lower()}"


def write_dataset(
    records: list[Record],
    audit: AuditReport,
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
        label = temporary / split / "labels" / f"{Path(name).stem}.txt"
        label_lines = (
            f"{annotation['class_id']} {annotation['x']:.6f} "
            f"{annotation['y']:.6f} {annotation['w']:.6f} "
            f"{annotation['h']:.6f}\n"
            for annotation in record["annotations"]
        )
        label.write_text("".join(label_lines), encoding="utf-8")
        counts = Counter(a["class_id"] for a in record["annotations"])
        rows.append(
            {
                "split": split,
                "source": record["source"],
                "old_split": record["old_split"],
                "group_id": record["group_id"],
                "output_image": relative_image.as_posix(),
                "source_image": record["image_path"],
                "sha256": record["sha256"],
                "phash": record["phash"],
                "width": record["width"],
                "height": record["height"],
                "is_negative": not record["annotations"],
                "instances_class_0": counts[0],
                "instances_class_1": counts[1],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(temporary / "manifest.csv", index=False)
    write_json(temporary / "audit_report.json", audit)
    yaml_config = {
        "path": str(output.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_text = yaml.safe_dump(yaml_config, allow_unicode=True, sort_keys=False)
    (temporary / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # Chỉ thay bản cũ sau khi toàn bộ kiểm tra đã đạt.
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


def validate_dataset(manifest: pd.DataFrame, output: Path) -> None:
    if manifest.empty:
        raise ValueError("Manifest không có dữ liệu")
    if manifest.groupby("group_id")["split"].nunique().max() != 1:
        raise ValueError("Có nhóm ảnh xuất hiện ở nhiều tập dữ liệu")
    if manifest.groupby("sha256")["split"].nunique().max() != 1:
        raise ValueError("Có ảnh trùng SHA-256 xuất hiện ở nhiều tập dữ liệu")
    for split in SPLITS:
        images = {path.stem for path in (output / split / "images").iterdir() if path.is_file()}
        labels = {path.stem for path in (output / split / "labels").glob("*.txt")}
        if images != labels:
            raise ValueError(f"Ảnh và nhãn không khớp ở tập {split}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chuẩn hóa và hợp nhất các dataset YOLO lá lúa")
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
    records, audit = collect_records(sources, not args.drop_negatives)
    records, dedup_audit = deduplicate_and_group(records, args.phash_distance)
    audit.update(dedup_audit)
    assign_splits(records)
    manifest = write_dataset(records, audit, args.output, overwrite=args.overwrite)
    summary = manifest.groupby("split").agg(images=("output_image", "count"),
                                             negatives=("is_negative", "sum"),
                                             bacterial_boxes=("instances_class_0", "sum"),
                                             brown_spot_boxes=("instances_class_1", "sum"))
    print(f"\nDataset đã tạo tại {args.output.resolve()}\n{summary}")
    print(f"Đã loại {audit['exact_duplicates_removed']} ảnh trùng tuyệt đối; "
          f"tạo {audit['near_duplicate_links']} liên kết gần trùng.")


if __name__ == "__main__":
    main()
