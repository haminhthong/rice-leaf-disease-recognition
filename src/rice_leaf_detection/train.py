import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from .constants import SEED
from .utils import configure_utf8_console, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv8 phát hiện bệnh lá lúa")
    parser.add_argument("--data", type=Path, default=Path("data/processed/rice_leaf_detection/data.yaml"))
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", type=Path, help="Đường dẫn last.pt để resume đầy đủ")
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
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

    seed_everything(SEED)
    device = (
        args.device
        if args.device is not None
        else ("0" if torch.cuda.is_available() else "cpu")
    )
    batch = args.batch or (16 if torch.cuda.is_available() else 4)
    if args.resume:
        if not args.resume.exists():
            raise FileNotFoundError(args.resume)
        model = YOLO(str(args.resume))
        model.train(resume=True)
    else:
        if not args.data.exists():
            raise FileNotFoundError(f"Không tìm thấy {args.data}. Hãy chạy bước prepare trước.")
        run_name = args.name or f"yolov8s_{time.strftime('%Y%m%d_%H%M%S')}"
        model = YOLO(args.model)
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            batch=batch,
            imgsz=args.imgsz,
            device=device,
            optimizer="AdamW",
            lr0=0.001,
            weight_decay=0.0005,
            patience=args.patience,
            seed=SEED,
            deterministic=True,
            workers=args.workers,
            val=True,
            save=True,
            save_period=10,
            plots=True,
            project=str(args.runs_dir),
            name=run_name,
            exist_ok=False,
        )
    run_dir = Path(model.trainer.save_dir)
    if not args.resume:
        metadata = {
            "run_name": run_dir.name,
            "data_yaml": str(args.data.resolve()),
            "seed": SEED,
            "epochs_requested": args.epochs,
            "batch_size": batch,
            "image_size": args.imgsz,
            "model": args.model,
            "device": str(device),
        }
        metadata_path = run_dir / "run_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Quá trình huấn luyện chưa tạo {best_weights}")
    print(f"Thư mục kết quả: {run_dir.resolve()}")
    print(f"Trọng số tốt nhất: {best_weights.resolve()}")


if __name__ == "__main__":
    main()
