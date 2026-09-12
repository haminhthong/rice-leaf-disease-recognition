"""Script đánh giá mô hình YOLOv8 trên tập Validation hoặc Test.

Sử dụng:
  python scripts/evaluate.py --weights runs/train/yolov8s_640/weights/best.pt --split val
  python scripts/evaluate.py --weights runs/train/yolov8s_640/weights/best.pt --split test
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rice_leaf_detection.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
