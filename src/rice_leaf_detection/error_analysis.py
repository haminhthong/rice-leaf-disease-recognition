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
    """Tính IoU của hai bounding box ở định dạng x1, y1, x2, y2."""
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def match_detections(
    ground_truth: list[LabeledBox],
    predictions: list[LabeledBox],
    iou_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], Counter]:
    """Ghép kết quả dự đoán với nhãn thật cùng lớp theo độ tin cậy giảm dần."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("Ngưỡng IoU phải nằm trong khoảng (0, 1]")
    matched_ground_truth: set[int] = set()
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for class_id, predicted_box, confidence in sorted(
        predictions, key=lambda item: item[2], reverse=True
    ):
        candidates = [
            (index, box_iou(predicted_box, true_box))
            for index, (true_class, true_box, _) in enumerate(ground_truth)
            if true_class == class_id and index not in matched_ground_truth
        ]
        best_index, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        if best_iou >= iou_threshold:
            matched_ground_truth.add(best_index)
            counts["true_positive"] += 1
        else:
            counts["false_positive"] += 1
            errors.append(
                {
                    "error_type": "false_positive",
                    "class_id": class_id,
                    "confidence": confidence,
                    "best_iou": best_iou,
                }
            )

    for index, (class_id, _, _) in enumerate(ground_truth):
        if index not in matched_ground_truth:
            counts["false_negative"] += 1
            errors.append(
                {
                    "error_type": "false_negative",
                    "class_id": class_id,
                    "confidence": None,
                    "best_iou": None,
                }
            )
    return errors, counts


def read_yolo_labels(path: Path, width: int, height: int) -> list[LabeledBox]:
    boxes: list[LabeledBox] = []
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
    """Chuẩn hóa giá trị boolean đọc từ CSV."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Không thể chuyển thành boolean: {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phân tích lỗi phát hiện theo ảnh và nguồn")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/rice_leaf_detection"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/error_analysis"))
    parser.add_argument("--confirm-final-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit("Chỉ phân tích test sau khi chốt mô hình bằng tập xác thực.")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence phải nằm trong khoảng [0, 1]")
    for path in (args.weights, args.dataset / "manifest.csv"):
        if not path.exists():
            raise FileNotFoundError(path)

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    manifest = pd.read_csv(args.dataset / "manifest.csv")
    manifest = manifest[manifest["split"] == args.split]
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()

    for record in manifest.itertuples(index=False):
        image_path = args.dataset / record.output_image
        label_path = args.dataset / args.split / "labels" / f"{image_path.stem}.txt"
        truth = read_yolo_labels(label_path, int(record.width), int(record.height))
        result = model.predict(
            source=str(image_path), conf=args.confidence, iou=0.7, verbose=False
        )[0]
        predictions: list[LabeledBox] = [
            (
                int(box.cls[0]),
                tuple(float(value) for value in box.xyxy[0].tolist()),
                float(box.conf[0]),
            )
            for box in result.boxes
        ]
        image_errors, counts = match_detections(truth, predictions, args.iou)
        totals.update(counts)
        for error in image_errors:
            rows.append(
                {
                    "image": record.output_image,
                    "source": record.source,
                    "is_negative": parse_boolean(record.is_negative),
                    **error,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    errors = pd.DataFrame(rows)
    errors.to_csv(args.output / f"{args.split}_errors.csv", index=False)
    negative_false_positives = 0
    if not errors.empty:
        negative_mask = errors["is_negative"].astype(str).str.lower().eq("true")
        false_positive_mask = errors["error_type"].eq("false_positive")
        negative_false_positives = int(
            errors.loc[negative_mask & false_positive_mask, "image"].nunique()
        )
    summary = {
        "split": args.split,
        "confidence": args.confidence,
        "iou_threshold": args.iou,
        **totals,
        "images": len(manifest),
        "negative_images_with_false_positive": negative_false_positives,
    }
    (args.output / f"{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
