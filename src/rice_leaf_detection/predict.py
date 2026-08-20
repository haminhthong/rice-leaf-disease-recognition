import argparse
from collections import Counter
from pathlib import Path

from ultralytics import YOLO

from .constants import CLASS_NAMES_VI
from .utils import configure_utf8_console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dự đoán bệnh trên ảnh, thư mục hoặc video")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    parser.add_argument("--save-txt", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf phải nằm trong khoảng [0, 1]")
    if not 0 <= args.iou <= 1:
        raise ValueError("--iou phải nằm trong khoảng [0, 1]")
    for path in (args.weights, args.source):
        if not path.exists():
            raise FileNotFoundError(path)
    model = YOLO(str(args.weights))
    results = model.predict(source=str(args.source), conf=args.conf, iou=args.iou,
                            save=True, save_txt=args.save_txt, project=str(args.output),
                            name="results", exist_ok=True, stream=True)
    counts: Counter[int] = Counter()
    total_detections = 0
    processed_results = 0
    source_paths: set[str] = set()
    save_dir: Path | None = None
    for result in results:
        processed_results += 1
        source_paths.add(str(result.path))
        total_detections += len(result.boxes)
        save_dir = Path(result.save_dir)
        for class_id in result.boxes.cls.cpu().tolist():
            counts[int(class_id)] += 1

    print(f"Số kết quả đã xử lý: {processed_results}")
    print(f"Số đường dẫn nguồn: {len(source_paths)}")
    print(f"Tổng số vùng phát hiện: {total_detections}")
    for class_id, count in sorted(counts.items()):
        class_name = CLASS_NAMES_VI.get(class_id)
        if class_name is None:
            if isinstance(model.names, dict):
                class_name = str(model.names.get(class_id, class_id))
            elif 0 <= class_id < len(model.names):
                class_name = str(model.names[class_id])
            else:
                class_name = str(class_id)
        print(f"- {class_name}: {count}")
    if save_dir is not None:
        print(f"Kết quả: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
