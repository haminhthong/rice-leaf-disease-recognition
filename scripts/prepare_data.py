"""Script chuẩn hóa và chia tập dữ liệu lá lúa (Data Preparation).

Sử dụng:
  python scripts/prepare_data.py --overwrite
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rice_leaf_detection.prepare import main  # noqa: E402

if __name__ == "__main__":
    main()
