"""Pipeline đánh giá hiệu năng mô hình (Model Evaluation Pipeline).

Module này tính toán các độ đo chuẩn trong Object Detection:
Precision, Recall, mAP@50 và mAP@50-95 trên tập xác thực (Validation) hoặc tập kiểm thử (Test).

Đặc biệt: Tập Test mặc định bị khóa và yêu cầu cờ `--confirm-final-test` để đảm bảo tuân thủ
đúng ML protocol, ngăn chặn việc sử dụng kết quả tập test để chọn hyperparameter (Data Leakage).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import load_config
from .constants import OUT_OF_SCOPE_NEGATIVE, TRUE_NEGATIVE
from .error_analysis import box_iou, read_yolo_labels
from .utils import configure_utf8_console


def calculate_image_level_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Tính recall theo ảnh và false-alarm rate theo loại negative.

    Mỗi row cần có ``truth_classes``, ``detected_classes``,
    ``annotation_status`` và ``accepted_detection``. Hàm thuần để có thể test
    mà không cần nạp YOLO.
    """
    classes = (0, 1)
    positive_total = {str(class_id): 0 for class_id in classes}
    positive_recalled = {str(class_id): 0 for class_id in classes}
    negative_total = {TRUE_NEGATIVE: 0, OUT_OF_SCOPE_NEGATIVE: 0}
    negative_false_alarm = {TRUE_NEGATIVE: 0, OUT_OF_SCOPE_NEGATIVE: 0}

    for row in rows:
        truth_classes = {int(value) for value in row.get("truth_classes", [])}
        detected_classes = {int(value) for value in row.get("detected_classes", [])}
        for class_id in truth_classes:
            key = str(class_id)
            if key not in positive_total:
                continue
            positive_total[key] += 1
            if class_id in detected_classes:
                positive_recalled[key] += 1
        status = row.get("annotation_status")
        if status in negative_total:
            negative_total[status] += 1
            if row.get("accepted_detection", False):
                negative_false_alarm[status] += 1

    target_recall = {
        key: {
            "positive_images": positive_total[key],
            "recalled_images": positive_recalled[key],
            "recall": (
                round(positive_recalled[key] / positive_total[key], 4)
                if positive_total[key]
                else None
            ),
        }
        for key in ("0", "1")
    }
    false_alarm_rate = {
        status: {
            "negative_images": negative_total[status],
            "images_with_accepted_detection": negative_false_alarm[status],
            "false_alarm_rate": (
                round(negative_false_alarm[status] / negative_total[status], 4)
                if negative_total[status]
                else None
            ),
        }
        for status in (TRUE_NEGATIVE, OUT_OF_SCOPE_NEGATIVE)
    }
    return {
        "image_target_recall": target_recall,
        "negative_benchmark": false_alarm_rate,
    }


def evaluate_image_level_metrics(
    model: Any,
    dataset_dir: Path,
    manifest: pd.DataFrame,
    split: str,
    review_threshold: float,
    accept_threshold: float,
) -> dict[str, Any]:
    """Chạy matching đơn giản trên từng ảnh để đo metric phục vụ scouting."""
    rows: list[dict[str, Any]] = []
    for record in manifest[manifest["split"] == split].itertuples(index=False):
        image_path = dataset_dir / record.output_image
        label_path = dataset_dir / split / "labels" / f"{image_path.stem}.txt"
        truth = read_yolo_labels(label_path, int(record.width), int(record.height))
        result = model.predict(
            source=str(image_path),
            conf=review_threshold,
            iou=0.7,
            verbose=False,
        )[0]
        detected_classes: set[int] = set()
        accepted_detection = False
        for box in result.boxes:
            score = float(box.conf[0])
            if score >= accept_threshold:
                detected_classes.add(int(box.cls[0]))
                accepted_detection = True
        recalled_classes = set()
        for true_class, true_box, _ in truth:
            for box in result.boxes:
                if (
                    int(box.cls[0]) == true_class
                    and float(box.conf[0]) >= accept_threshold
                    and box_iou(
                        tuple(float(value) for value in box.xyxy[0].tolist()),
                        true_box,
                    )
                    >= 0.5
                ):
                    recalled_classes.add(true_class)
                    break
        rows.append(
            {
                "truth_classes": [class_id for class_id, _, _ in truth],
                "detected_classes": recalled_classes,
                "annotation_status": (
                    getattr(record, "annotation_status", None)
                    or (
                        TRUE_NEGATIVE
                        if str(getattr(record, "is_negative", "False")).lower() == "true"
                        else "TARGET_POSITIVE"
                    )
                ),
                "accepted_detection": accepted_detection,
            }
        )
    return calculate_image_level_metrics(rows)


def parse_args() -> argparse.Namespace:
    """Phân tích các tham số truyền từ giao diện dòng lệnh (CLI)."""
    parser = argparse.ArgumentParser(description="Đánh giá YOLOv8 trên tập xác thực hoặc test")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn trọng số best.pt")
    parser.add_argument("--data", type=Path, help="Đường dẫn file data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="Phân tập đánh giá")
    parser.add_argument("--imgsz", type=int, help="Kích thước ảnh")
    parser.add_argument("--batch", type=int, help="Kích thước batch")
    parser.add_argument("--device", default=None, help="Thiết bị tính toán")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/evaluate"),
        help="Thư mục xuất báo cáo",
    )

    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Xác nhận mở khóa chạy đánh giá trên tập Test",
    )
    return parser.parse_args()


def main() -> None:
    """Hàm thực thi chính của pipeline đánh giá."""
    import torch
    from ultralytics import YOLO

    configure_utf8_console()
    args = parse_args()
    config = load_config(args.config)

    args.data = args.data or config.data.yaml
    args.imgsz = args.imgsz if args.imgsz is not None else config.data.image_size

    if args.imgsz <= 0:
        raise ValueError("--imgsz phải lớn hơn 0")
    if args.batch is not None and args.batch <= 0:
        raise ValueError("--batch phải lớn hơn 0")

    # Kiểm tra khóa tập Test để tuân thủ ML protocol
    if args.split == "test" and not args.confirm_final_test:
        raise SystemExit(
            "Tập test đang được khóa. Chỉ thêm --confirm-final-test sau khi "
            "đã chọn mô hình bằng tập xác thực."
        )

    for path in (args.weights, args.data):
        if not path.exists():
            raise FileNotFoundError(path)

    device = (
        args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    )
    batch = args.batch or (
        config.training.batch_gpu if torch.cuda.is_available() else config.training.batch_cpu
    )

    # Chạy validation bằng Ultralytics YOLO API
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        conf=0.001,
        iou=0.7,
        plots=True,
        augment=False,
        project=str(args.output),
        name=f"{args.weights.parent.parent.name}_{args.split}",
        exist_ok=True,
    )

    summary = {
        "run_name": args.weights.parent.parent.name,
        "weights": str(args.weights.resolve()),
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }

    # Metric theo ảnh giúp phản ánh đúng use case scouting: chỉ cần ít nhất
    # một detection đúng trên ảnh bệnh, đồng thời phải đo false alarm riêng
    # cho healthy và bệnh ngoài phạm vi.
    manifest_path = args.data.parent / "manifest.csv"
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        summary["image_level"] = evaluate_image_level_metrics(
            model=model,
            dataset_dir=args.data.parent,
            manifest=manifest,
            split=args.split,
            review_threshold=config.policy.review_threshold,
            accept_threshold=config.policy.accept_threshold,
        )

    # Tổng hợp metric chi tiết theo từng lớp bệnh
    rows = []
    for class_id in sorted(metrics.names):
        precision, recall, ap50, ap50_95 = metrics.box.class_result(class_id)
        rows.append(
            {
                "class_id": class_id,
                "class_name": metrics.names[class_id],
                "precision": float(precision),
                "recall": float(recall),
                "AP50": float(ap50),
                "AP50-95": float(ap50_95),
            }
        )

    save_dir = Path(metrics.save_dir)
    (save_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(save_dir / "per_class_metrics.csv", index=False)

    # Cập nhật lịch sử thí nghiệm tổng hợp vào experiments.csv
    args.output.mkdir(parents=True, exist_ok=True)
    experiment_log = args.output / "experiments.csv"
    current = pd.DataFrame([summary])
    if experiment_log.exists():
        history = pd.read_csv(experiment_log)
        history = history[
            ~((history["run_name"] == summary["run_name"]) & (history["split"] == summary["split"]))
        ]
        current = pd.concat([history, current], ignore_index=True)
    current.to_csv(experiment_log, index=False)

    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
