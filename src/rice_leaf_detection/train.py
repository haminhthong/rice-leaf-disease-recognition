"""Huấn luyện mô hình YOLOv8 phát hiện tổn thương bệnh lá lúa (Training Pipeline)."""

import argparse
import json
import sys
import time
from pathlib import Path

from ultralytics import YOLO

from .config import load_config
from .utils import configure_utf8_console, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv8 phát hiện bệnh lá lúa")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data", type=Path, help="Đường dẫn file data.yaml")
    parser.add_argument("--model", help="Trọng số khởi tạo (vd: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Kích thước ảnh đầu vào (pixels)")
    parser.add_argument("--patience", type=int, help="Early stopping patience (epochs)")
    parser.add_argument("--device", default=None, help="Thiết bị tính toán (0 cho GPU, cpu)")
    parser.add_argument("--workers", type=int, help="Số worker DataLoader")
    parser.add_argument("--runs-dir", type=Path, help="Thư mục lưu trữ kết quả")
    parser.add_argument("--name", default=None, help="Tên đợt huấn luyện")
    parser.add_argument("--resume", type=Path, help="Đường dẫn last.pt để tiếp tục huấn luyện")
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    config = load_config(args.config)
    import torch

    args.data = args.data or config.data.yaml
    args.model = args.model or config.model.weights
    args.epochs = args.epochs if args.epochs is not None else config.training.epochs
    args.imgsz = args.imgsz if args.imgsz is not None else config.data.image_size
    args.patience = args.patience if args.patience is not None else config.training.patience
    args.workers = (
        args.workers
        if args.workers is not None
        else (0 if sys.platform == "win32" else config.training.workers)
    )
    args.runs_dir = args.runs_dir or config.project.runs_dir
    seed = config.project.seed

    if args.epochs <= 0:
        raise ValueError("--epochs phải lớn hơn 0")
    if args.batch is not None and args.batch <= 0:
        raise ValueError("--batch phải lớn hơn 0")
    if args.imgsz <= 0:
        raise ValueError("--imgsz phải lớn hơn 0")
    if args.patience < 0:
        raise ValueError("--patience không được âm")
    if args.workers < 0:
        raise ValueError("--workers không được âm")

    seed_everything(seed)

    device = (
        args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    )
    batch = args.batch or int(
        config.training.batch_gpu if torch.cuda.is_available() else config.training.batch_cpu
    )

    if args.resume:
        if not args.resume.exists():
            raise FileNotFoundError(args.resume)
        model = YOLO(str(args.resume))
        model.train(resume=True)
    else:
        if not args.data.exists():
            raise FileNotFoundError(f"Không tìm thấy {args.data}. Hãy chạy prepare_data.py trước.")
        run_name = args.name or (
            f"{config.model.architecture}_640_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        model = YOLO(args.model)
        aug = config.training.augmentation
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            batch=batch,
            imgsz=args.imgsz,
            device=device,
            optimizer=config.training.optimizer,
            lr0=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            patience=args.patience,
            seed=seed,
            deterministic=True,
            workers=args.workers,
            # Augmentation chỉ áp dụng cho tập train; val/test tất định
            hsv_h=aug.hsv_h,
            hsv_s=aug.hsv_s,
            hsv_v=aug.hsv_v,
            degrees=aug.degrees,
            translate=aug.translate,
            scale=aug.scale,
            fliplr=aug.fliplr,
            flipud=aug.flipud,
            mosaic=aug.mosaic,
            mixup=aug.mixup,
            close_mosaic=aug.close_mosaic,
            val=True,
            save=True,
            plots=True,
            project=str(args.runs_dir),
            name=run_name,
            exist_ok=False,
        )

    run_dir = Path(model.trainer.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Quá trình huấn luyện chưa tạo {best_weights}")

    metadata = {
        "run_name": run_dir.name,
        "data_yaml": str(args.data.resolve()),
        "epochs": args.epochs,
        "batch_size": batch,
        "image_size": args.imgsz,
        "model": args.model,
        "device": str(device),
        "best_weights": str(best_weights.resolve()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nHuấn luyện hoàn tất!")
    print(f"Thư mục kết quả: {run_dir.resolve()}")
    print(f"Trọng số tốt nhất: {best_weights.resolve()}")


if __name__ == "__main__":
    main()
