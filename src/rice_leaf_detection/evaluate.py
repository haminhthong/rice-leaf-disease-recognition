"""Pipeline đánh giá hiệu năng mô hình (Model Evaluation Pipeline).

Module này tính toán các độ đo chuẩn trong Object Detection:
Precision, Recall, mAP@50 và mAP@50-95 trên tập xác thực (Validation) hoặc tập kiểm thử (Test).

Đặc biệt: Tập Test mặc định bị khóa và yêu cầu cờ `--confirm-final-test` để đảm bảo tuân thủ
đúng ML protocol, ngăn chặn việc sử dụng kết quả tập test để chọn hyperparameter (Data Leakage).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from ultralytics import YOLO

from .config import load_config
from .utils import configure_utf8_console


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
    metrics = YOLO(str(args.weights)).val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        conf=0.001,
        iou=0.7,
        plots=True,
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
