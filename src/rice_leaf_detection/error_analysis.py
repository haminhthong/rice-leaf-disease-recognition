"""Phân tích lỗi mô hình chi tiết (Error Analysis Pipeline).

Ghép nối kết quả dự đoán với nhãn Ground Truth để phân loại lỗi:
- True Positive (TP)
- False Positive (FP): background, localization, classification confusion, duplicate
- False Negative (FN): bỏ sót tổn thương
- Phân bố theo kích thước tổn thương: Small (< 0.05), Medium (0.05 - 0.20), Large (> 0.20)
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import configure_utf8_console

Box = tuple[float, float, float, float]
LabeledBox = tuple[int, Box, float]


def box_iou(left: Box, right: Box) -> float:
    """Tính IoU của hai bounding box ở định dạng (x1, y1, x2, y2)."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def classify_lesion_size(box: Box, width: int, height: int) -> str:
    """Phân loại kích thước tổn thương dựa trên tỷ lệ diện tích box so với ảnh."""
    if width <= 0 or height <= 0:
        return "medium"
    area_fraction = ((box[2] - box[0]) * (box[3] - box[1])) / (width * height)
    if area_fraction < 0.05:
        return "small"
    elif area_fraction <= 0.20:
        return "medium"
    return "large"


def match_detections(
    ground_truth: list[LabeledBox],
    predictions: list[LabeledBox],
    iou_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], Counter]:
    """Ghép kết quả dự đoán với nhãn thật theo độ tin cậy giảm dần."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("Ngưỡng IoU phải nằm trong khoảng (0, 1]")
    matched_ground_truth: set[int] = set()
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for class_id, predicted_box, confidence in sorted(
        predictions, key=lambda item: item[2], reverse=True
    ):
        same_class_candidates = [
            (index, box_iou(predicted_box, true_box))
            for index, (true_class, true_box, _) in enumerate(ground_truth)
            if true_class == class_id
        ]
        best_same_idx, best_same_iou = max(
            same_class_candidates, key=lambda item: item[1], default=(-1, 0.0)
        )
        unmatched_same_candidates = [
            (index, overlap)
            for index, overlap in same_class_candidates
            if index not in matched_ground_truth
        ]
        best_unmatched_idx, best_unmatched_iou = max(
            unmatched_same_candidates,
            key=lambda item: item[1],
            default=(-1, 0.0),
        )

        all_candidates = [
            (index, box_iou(predicted_box, true_box), true_class)
            for index, (true_class, true_box, _) in enumerate(ground_truth)
        ]
        _, best_any_iou, best_any_class = max(
            all_candidates, key=lambda item: item[1], default=(-1, 0.0, -1)
        )

        if best_unmatched_iou >= iou_threshold:
            matched_ground_truth.add(best_unmatched_idx)
            counts["true_positive"] += 1
        elif best_same_iou >= iou_threshold:
            if best_same_idx in matched_ground_truth:
                counts["false_positive"] += 1
                counts["duplicate_detection"] += 1
                errors.append(
                    {
                        "error_type": "false_positive",
                        "detailed_error_type": "duplicate_detection",
                        "class_id": class_id,
                        "confidence": confidence,
                        "best_iou": best_same_iou,
                    }
                )
            else:
                counts["false_positive"] += 1
        else:
            counts["false_positive"] += 1
            if best_any_iou >= iou_threshold and best_any_class != class_id:
                counts["classification_confusion"] += 1
                errors.append(
                    {
                        "error_type": "false_positive",
                        "detailed_error_type": "classification_confusion",
                        "class_id": class_id,
                        "true_class": best_any_class,
                        "confidence": confidence,
                        "best_iou": best_any_iou,
                    }
                )
            elif 0.1 <= best_same_iou < iou_threshold:
                counts["localization_error"] += 1
                errors.append(
                    {
                        "error_type": "false_positive",
                        "detailed_error_type": "localization_error",
                        "class_id": class_id,
                        "confidence": confidence,
                        "best_iou": best_same_iou,
                    }
                )
            else:
                counts["false_positive_background"] += 1
                errors.append(
                    {
                        "error_type": "false_positive",
                        "detailed_error_type": "false_positive_background",
                        "class_id": class_id,
                        "confidence": confidence,
                        "best_iou": best_same_iou,
                    }
                )

    for index, (class_id, _, _) in enumerate(ground_truth):
        if index not in matched_ground_truth:
            counts["false_negative"] += 1
            errors.append(
                {
                    "error_type": "false_negative",
                    "detailed_error_type": "false_negative_missed_lesion",
                    "class_id": class_id,
                    "confidence": None,
                    "best_iou": None,
                    "gt_index": index,
                }
            )
    return errors, counts


def read_yolo_labels(path: Path, width: int, height: int) -> list[LabeledBox]:
    boxes: list[LabeledBox] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, box_width, box_height = map(float, line.split())
        boxes.append(
            (
                int(class_id),
                (
                    (x - box_width / 2) * width,
                    (y - box_height / 2) * height,
                    (x + box_width / 2) * width,
                    (y + box_height / 2) * height,
                ),
                1.0,
            )
        )
    return boxes


def parse_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Không thể chuyển thành boolean: {value!r}")


def run_error_analysis(
    weights_path: Path | str,
    dataset_dir: Path | str = Path("data/processed/rice_leaf_detection"),
    split: str = "val",
    confidence: float = 0.45,
    iou: float = 0.5,
    image_size: int = 640,
    output_dir: Path | str = Path("reports/error_analysis"),
) -> dict[str, Any]:
    """Phân tích lỗi mô hình theo lát cắt kích thước tổn thương và phân loại lỗi."""
    from ultralytics import YOLO

    weights_path = Path(weights_path)
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    manifest_path = dataset_dir / "manifest.csv"
    if not weights_path.exists():
        raise FileNotFoundError(f"Không tìm thấy trọng số: {weights_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest: {manifest_path}")

    model = YOLO(str(weights_path))
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["split"] == split]

    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    lesion_size_totals = {"small": 0, "medium": 0, "large": 0}
    lesion_size_recalled = {"small": 0, "medium": 0, "large": 0}

    for record in manifest.itertuples(index=False):
        image_path = dataset_dir / record.output_image
        label_path = dataset_dir / split / "labels" / f"{image_path.stem}.txt"
        truth = read_yolo_labels(label_path, int(record.width), int(record.height))
        result = model.predict(
            source=str(image_path),
            imgsz=image_size,
            conf=confidence,
            iou=0.7,
            verbose=False,
        )[0]
        predictions: list[LabeledBox] = [
            (
                int(box.cls[0]),
                tuple(float(v) for v in box.xyxy[0].tolist()),
                float(box.conf[0]),
            )
            for box in result.boxes
        ]
        image_errors, counts = match_detections(truth, predictions, iou)
        totals.update(counts)

        unmatched_indices = {e["gt_index"] for e in image_errors if "gt_index" in e}
        for gt_idx, (_, gt_box, _) in enumerate(truth):
            size_cat = classify_lesion_size(gt_box, int(record.width), int(record.height))
            lesion_size_totals[size_cat] += 1
            if gt_idx not in unmatched_indices:
                lesion_size_recalled[size_cat] += 1

        for err in image_errors:
            rows.append(
                {
                    "image": record.output_image,
                    "source": record.source,
                    "is_negative": parse_boolean(record.is_negative),
                    **err,
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / f"{split}_errors.csv", index=False)

    summary = {
        "split": split,
        "confidence": confidence,
        "iou_threshold": iou,
        "true_positive": totals.get("true_positive", 0),
        "false_positive": totals.get("false_positive", 0),
        "false_negative": totals.get("false_negative", 0),
        "error_taxonomy": {
            "false_negative_missed_lesion": totals.get("false_negative", 0),
            "false_positive_background": totals.get("false_positive_background", 0),
            "localization_error": totals.get("localization_error", 0),
            "classification_confusion": totals.get("classification_confusion", 0),
            "duplicate_detection": totals.get("duplicate_detection", 0),
        },
        "lesion_size_recall": {
            cat: {
                "total": lesion_size_totals[cat],
                "recalled": lesion_size_recalled[cat],
                "recall": (
                    round(lesion_size_recalled[cat] / lesion_size_totals[cat], 4)
                    if lesion_size_totals[cat] > 0
                    else None
                ),
            }
            for cat in ("small", "medium", "large")
        },
    }

    (output_dir / "error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phân tích lỗi phát hiện của mô hình")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/rice_leaf_detection"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=Path, default=Path("reports/error_analysis"))
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    summary = run_error_analysis(
        weights_path=args.weights,
        dataset_dir=args.dataset,
        split=args.split,
        confidence=args.confidence,
        iou=args.iou,
        image_size=args.imgsz,
        output_dir=args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
