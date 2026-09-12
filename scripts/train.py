"""Script huấn luyện mô hình YOLOv8 phát hiện bệnh lá lúa.

Sử dụng:
  python scripts/train.py --epochs 50 --batch 16
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rice_leaf_detection.train import main  # noqa: E402

if __name__ == "__main__":
    main()
