"""Công cụ suy luận từ giao diện dòng lệnh (CLI Prediction Tool)."""

import argparse
from collections import Counter
from pathlib import Path

from .config import load_config
from .constants import CLASS_NAMES_VI
from .inference import RiceLeafDetector
from .utils import configure_utf8_console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dự đoán bệnh trên ảnh hoặc thư mục ảnh")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--weights", type=Path, required=True, help="Đường dẫn file best.pt")
    parser.add_argument("--source", type=Path, required=True, help="File ảnh hoặc thư mục")
    parser.add_argument("--conf", type=float, help="Ngưỡng tin cậy (Confidence threshold)")
    parser.add_argument("--iou", type=float, help="Ngưỡng NMS IoU")
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    parser.add_argument("--save-txt", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    config = load_config(args.config)
    conf = args.conf if args.conf is not None else config.inference.confidence
    iou = args.iou if args.iou is not None else config.inference.iou

    if not 0 <= conf <= 1:
        raise ValueError("--conf phải nằm trong khoảng [0, 1]")
    if not 0 <= iou <= 1:
        raise ValueError("--iou phải nằm trong khoảng [0, 1]")
    for p in (args.weights, args.source):
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy: {p}")

    detector = RiceLeafDetector(
        weights=args.weights,
        image_size=config.data.image_size,
        confidence=conf,
        iou=iou,
    )

    results = detector.model.predict(
        source=str(args.source),
        imgsz=config.data.image_size,
        conf=conf,
        iou=iou,
        save=True,
        save_txt=args.save_txt,
        project=str(args.output),
        name="results",
        exist_ok=True,
        stream=True,
    )

    counts: Counter[int] = Counter()
    total_detections = 0
    processed_count = 0
    save_dir: Path | None = None

    for res in results:
        processed_count += 1
        save_dir = Path(res.save_dir)
        for box in res.boxes:
            class_id = int(box.cls[0])
            counts[class_id] += 1
            total_detections += 1

    print(f"\nSố ảnh đã xử lý: {processed_count}")
    print(f"Tổng số vùng phát hiện: {total_detections}")
    if total_detections:
        print("Trạng thái: DETECTED")
    else:
        print("Trạng thái: NO_SYMPTOM_DETECTED")

    for class_id, count in sorted(counts.items()):
        class_name = CLASS_NAMES_VI.get(class_id, f"Lớp {class_id}")
        print(f"- {class_name}: {count}")

    if save_dir is not None:
        print(f"Kết quả lưu tại: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
