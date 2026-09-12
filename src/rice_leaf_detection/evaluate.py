"""Đánh giá hiệu năng mô hình YOLOv8 trên tập Validation hoặc Test (Evaluation Pipeline).

Tính toán 4 độ đo chuẩn trong Object Detection:
Precision, Recall, mAP@0.5 và mAP@0.5:0.95, cùng báo cáo chi tiết theo từng lớp bệnh.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .config import load_config
from .utils import configure_utf8_console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Đánh giá YOLOv8 trên tập Validation hoặc Test")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn file best.pt")
    parser.add_argument("--data", type=Path, help="Đường dẫn file data.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val", help="Phân tập đánh giá")
    parser.add_argument("--imgsz", type=int, help="Kích thước ảnh")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--device", default=None, help="Thiết bị tính toán")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/evaluate"),
        help="Thư mục xuất báo cáo",
    )
    return parser.parse_args()


def main() -> None:
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

    for p in (args.weights, args.data):
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy: {p}")

    device = (
        args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    )
    batch = args.batch or (
        config.training.batch_gpu if torch.cuda.is_available() else config.training.batch_cpu
    )

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
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    per_class_rows = []
    for class_id in sorted(metrics.names):
        precision, recall, ap50, ap50_95 = metrics.box.class_result(class_id)
        per_class_rows.append(
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
    pd.DataFrame(per_class_rows).to_csv(save_dir / "per_class_metrics.csv", index=False)

    print("\n" + "=" * 50)
    print(f"KẾT QUẢ ĐÁNH GIÁ TẬP {args.split.upper()}:")
    print(f"- Precision : {summary['precision']:.4f}")
    print(f"- Recall    : {summary['recall']:.4f}")
    print(f"- mAP@0.5   : {summary['mAP50']:.4f}")
    print(f"- mAP@0.5:95: {summary['mAP50-95']:.4f}")
    print("=" * 50)
    print(pd.DataFrame(per_class_rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
