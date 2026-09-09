"""Script tiện ích chạy Pipeline Nhận Diện Bệnh Lá Lúa (End-to-End MLOps Pipeline Runner).

Cách sử dụng:
  # 1. In sơ đồ DAG và kiểm tra điều kiện tiên quyết:
  python scripts/run_pipeline.py --dry-run

  # 2. Chạy quy trình chuẩn hóa và chia dữ liệu (Data Engineering):
  python scripts/run_pipeline.py --stage data

  # 3. Chạy toàn bộ Pipeline từ dữ liệu thô đến export artifact:
  python scripts/run_pipeline.py --stage all --epochs 5

  # 4. Đánh giá tập Test bị khóa sau khi chốt model và policy:
  python scripts/run_pipeline.py --stage test --confirm-final-test
"""

import sys
from pathlib import Path

# Đảm bảo import được module rice_leaf_detection khi chạy trực tiếp từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from rice_leaf_detection.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
