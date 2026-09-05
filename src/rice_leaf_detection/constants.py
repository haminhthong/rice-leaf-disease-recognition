"""Hằng số hệ thống cho dự án Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Recognition).

Module này định nghĩa các hằng số dùng chung trong toàn bộ pipeline:
danh sách tên lớp mục tiêu, tỷ lệ chia tập dữ liệu train/val/test,
seed cố định và các định dạng file ảnh được hỗ trợ.
"""

from pathlib import Path

# Hạt giống ngẫu nhiên dùng chung để các lần chạy có thể tái lập.
SEED: int = 42

# Tên lớp theo định dạng nhãn YOLO.
CLASS_NAMES: list[str] = ["Bacterial_Leaf_Blight", "Brown_Spot"]

# Tên tiếng Việt dùng khi hiển thị trên API và giao diện.
CLASS_NAMES_VI: dict[int, str] = {
    0: "Bạc lá lúa",
    1: "Đốm nâu",
}

# Các phần mở rộng ảnh được pipeline dữ liệu chấp nhận.
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Tên ba tập dữ liệu của thí nghiệm.
SPLITS: tuple[str, str, str] = ("train", "val", "test")

# Tỷ lệ mục tiêu; số lượng thực tế còn phụ thuộc vào các nhóm ảnh độc lập.
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

# Hai tệp dữ liệu nguồn được dùng khi không truyền --archives.
DEFAULT_ARCHIVES: tuple[Path, Path] = (
    Path("RiceLeafAnnotatedDataset.zip"),
    Path("dataset1.zip"),
)
