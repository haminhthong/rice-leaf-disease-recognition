"""Công cụ suy luận từ giao diện dòng lệnh (CLI Prediction Tool).

Cho phép người dùng thực hiện dự đoán bệnh lá lúa trên một ảnh lẻ, một thư mục chứa nhiều ảnh
hoặc một file video truyền trực tiếp từ dòng lệnh (`rice-predict`).
"""

import argparse
from collections import Counter
from pathlib import Path

from .config import load_config
from .constants import CLASS_NAMES_VI
from .inference import (
    DetectionPolicy,
    RawDetection,
    RiceLeafDetector,
    load_detection_policy,
)
from .utils import configure_utf8_console


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dự đoán bệnh trên ảnh, thư mục hoặc video")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--output", type=Path, default=Path("runs/predict"))
    parser.add_argument("--save-txt", action="store_true")
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    config = load_config(args.config)
    args.conf = args.conf if args.conf is not None else config.inference.candidate_confidence
    args.iou = args.iou if args.iou is not None else config.inference.iou
    if not 0 <= args.conf <= 1:
        raise ValueError("--conf phải nằm trong khoảng [0, 1]")
    if not 0 <= args.iou <= 1:
        raise ValueError("--iou phải nằm trong khoảng [0, 1]")
    for path in (args.weights, args.source):
        if not path.exists():
            raise FileNotFoundError(path)
    configured_policy = DetectionPolicy(
        review_threshold=config.policy.review_threshold,
        accept_threshold=config.policy.accept_threshold,
    )
    policy = load_detection_policy(
        args.weights.parent / "detection_policy.json",
        fallback=configured_policy,
    )
    detector = RiceLeafDetector(
        weights=args.weights,
        image_size=config.data.image_size,
        confidence=args.conf,
        iou=args.iou,
        policy=policy,
    )
    results = detector.model.predict(
        source=str(args.source),
        imgsz=config.data.image_size,
        # Lấy đủ candidate để policy có thể giữ lại vùng review.
        conf=min(args.conf, policy.review_threshold),
        iou=args.iou,
        save=True,
        save_txt=args.save_txt,
        project=str(args.output),
        name="results",
        exist_ok=True,
        stream=True,
    )
    counts: Counter[int] = Counter()
    total_detections = 0
    accepted_detections = 0
    review_detections = 0
    processed_results = 0
    source_paths: set[str] = set()
    save_dir: Path | None = None
    for result in results:
        processed_results += 1
        source_paths.add(str(result.path))
        save_dir = Path(result.save_dir)
        candidates = [
            RawDetection(
                class_id=int(box.cls[0]),
                class_name=str(detector.model.names[int(box.cls[0])]),
                score=float(box.conf[0]),
                box_xyxy=tuple(float(value) for value in box.xyxy[0].tolist()),
            )
            for box in result.boxes
        ]
        decisions = policy.apply(candidates)
        total_detections += len(decisions)
        accepted_detections += sum(d.decision == "accepted" for d in decisions)
        review_detections += sum(d.decision == "review" for d in decisions)
        for detection in decisions:
            if detection.decision == "accepted":
                counts[detection.class_id] += 1

    print(f"Số kết quả đã xử lý: {processed_results}")
    print(f"Số đường dẫn nguồn: {len(source_paths)}")
    print(f"Tổng số vùng phát hiện: {total_detections}")
    if accepted_detections:
        print(f"Trạng thái: DETECTED ({accepted_detections} vùng được chấp nhận)")
    elif review_detections:
        print(f"Trạng thái: REVIEW_REQUIRED ({review_detections} vùng cần kiểm tra)")
    else:
        print("Trạng thái: NO_SUPPORTED_SYMPTOM_DETECTED")
        print("Lưu ý: không đồng nghĩa với việc lá khỏe mạnh.")
    for class_id, count in sorted(counts.items()):
        class_name = CLASS_NAMES_VI.get(class_id)
        if class_name is None:
            if isinstance(detector.model.names, dict):
                class_name = str(detector.model.names.get(class_id, class_id))
            elif 0 <= class_id < len(detector.model.names):
                class_name = str(detector.model.names[class_id])
            else:
                class_name = str(class_id)
        print(f"- {class_name}: {count}")
    if save_dir is not None:
        print(f"Kết quả: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
