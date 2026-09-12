"""Script suy luận phát hiện bệnh lá lúa trên ảnh hoặc thư mục ảnh.

Sử dụng:
  python scripts/predict.py --weights best.pt --source data/sample/sample.jpg
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rice_leaf_detection.predict import main  # noqa: E402

if __name__ == "__main__":
    main()
