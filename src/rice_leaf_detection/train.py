"""Pipeline huấn luyện mô hình YOLOv8 phát hiện bệnh lá lúa (Training Pipeline).

Module này nhận các tham số cấu hình từ file YAML (hoặc các cờ dòng lệnh CLI),
cố định seed tái lập, tự động phát hiện thiết bị phần cứng (GPU/CPU), khởi tạo mô hình YOLOv8
và lưu vết metadata thí nghiệm sau khi huấn luyện thành công.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO

from .config import load_config
from .utils import configure_utf8_console, seed_everything


def parse_args() -> argparse.Namespace:
    """Phân tích các tham số truyền từ giao diện dòng lệnh (CLI)."""
    parser = argparse.ArgumentParser(description="Huấn luyện YOLOv8 phát hiện bệnh lá lúa")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data", type=Path, help="Đường dẫn file data.yaml")
    parser.add_argument("--model", help="Tên hoặc đường dẫn trọng số mô hình gốc (vd: yolov8s.pt)")
    parser.add_argument("--epochs", type=int, help="Số lượng epoch huấn luyện")
    parser.add_argument("--batch", type=int, help="Kích thước batch (Batch size)")
    parser.add_argument("--imgsz", type=int, help="Kích thước ảnh đầu vào (pixels)")
    parser.add_argument("--patience", type=int, help="Early stopping patience (epochs)")
    parser.add_argument("--device", default=None, help="Thiết bị tính toán (0 cho GPU, cpu)")

    parser.add_argument("--workers", type=int, help="Số lượng worker DataLoader")
    parser.add_argument("--runs-dir", type=Path, help="Thư mục lưu trữ kết quả thí nghiệm")
    parser.add_argument("--name", default=None, help="Tên đợt huấn luyện (Run name)")
    parser.add_argument("--resume", type=Path, help="Đường dẫn last.pt để tiếp tục huấn luyện")
    return parser.parse_args()


def main() -> None:
    """Hàm thực thi chính của pipeline huấn luyện."""
    configure_utf8_console()
    args = parse_args()
    config = load_config(args.config)

    # Ưu tiên các tham số truyền trực tiếp từ CLI, nếu không dùng từ file cấu hình YAML
    args.data = args.data or config.data.yaml
    args.model = args.model or config.model.weights
    args.epochs = args.epochs if args.epochs is not None else config.training.epochs
    args.imgsz = args.imgsz if args.imgsz is not None else config.data.image_size
    args.patience = (
        args.patience
        if args.patience is not None
        else config.training.patience
    )
    args.workers = (
        args.workers
        if args.workers is not None
        else config.training.workers
    )
    args.runs_dir = args.runs_dir or config.project.runs_dir
    seed = config.project.seed

    # Kiểm tra ràng buộc giá trị hợp lệ của tham số
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

    # Cố định ngẫu nhiên seed để đảm bảo tính tái lập
    seed_everything(seed)

    # Tự động cấu hình GPU CUDA nếu khả dụng, ngược lại dùng CPU
    device = (
        args.device
        if args.device is not None
        else ("0" if torch.cuda.is_available() else "cpu")
    )
    batch = args.batch or int(
        config.training.batch_gpu
        if torch.cuda.is_available()
        else config.training.batch_cpu
    )

    if args.resume:
        if not args.resume.exists():
            raise FileNotFoundError(args.resume)
        model = YOLO(str(args.resume))
        model.train(resume=True)
    else:
        if not args.data.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {args.data}. Hãy chạy lệnh rice-prepare trước."
            )
        run_name = args.name or f"yolov8s_{time.strftime('%Y%m%d_%H%M%S')}"
        model = YOLO(args.model)
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
            "seed": seed,
            "config": str(args.config.resolve()),
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

