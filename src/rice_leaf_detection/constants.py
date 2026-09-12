"""Hằng số hệ thống cho dự án Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Recognition).

Module này định nghĩa các hằng số dùng chung:
danh sách tên lớp mục tiêu, tỷ lệ chia tập dữ liệu train/val/test,
seed cố định và các định dạng file ảnh được hỗ trợ.
"""

from pathlib import Path
from typing import Literal

# Hạt giống ngẫu nhiên dùng chung để các lần chạy có thể tái lập.
SEED: int = 42

# Tên lớp theo định dạng nhãn YOLO.
CLASS_NAMES: list[str] = ["Bacterial_Leaf_Blight", "Brown_Spot"]

# Tên tiếng Việt dùng khi hiển thị trên API và giao diện.
CLASS_NAMES_VI: dict[int, str] = {
    0: "Bạc lá lúa",
    1: "Đốm nâu",
}

# Trạng thái annotation chuẩn hóa:
# - valid: Ảnh có nhãn hợp lệ chứa ít nhất 1 lesion thuộc 2 bệnh mục tiêu.
# - negative: Ảnh negative hợp lệ (không chứa tổn thương mục tiêu).
# - invalid: Ảnh thiếu file nhãn hoặc tọa độ nhãn lỗi -> loại bỏ (không được coi là negative).
AnnotationStatus = Literal["valid", "negative", "invalid"]

STATUS_VALID: AnnotationStatus = "valid"
STATUS_NEGATIVE: AnnotationStatus = "negative"
STATUS_INVALID: AnnotationStatus = "invalid"

# Các phần mở rộng ảnh được chấp nhận.
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Tên ba tập dữ liệu.
SPLITS: tuple[str, str, str] = ("train", "val", "test")

# Tỷ lệ mục tiêu chia tập.
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

# Hai tệp dữ liệu nguồn mặc định.
DEFAULT_ARCHIVES: tuple[Path, Path] = (
    Path("RiceLeafAnnotatedDataset.zip"),
    Path("dataset1.zip"),
)
