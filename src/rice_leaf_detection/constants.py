"""Hằng số hệ thống cho dự án Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Recognition).

Module này định nghĩa các hằng số dùng chung trong toàn bộ pipeline:
danh sách tên lớp mục tiêu, tỷ lệ chia tập dữ liệu train/val/test,
seed cố định và các định dạng file ảnh được hỗ trợ.
"""

from pathlib import Path

# Cố định ngẫu nhiên seed để đảm bảo tính tái lập (reproducibility)
SEED: int = 42

# Danh sách tên các lớp bệnh mục tiêu (Chuẩn tiếng Anh trong YOLO dataset)
CLASS_NAMES: list[str] = ["Bacterial_Leaf_Blight", "Brown_Spot"]

# Ánh xạ tên lớp mục tiêu sang Tiếng Việt hiển thị trên API / Web Dashboard
CLASS_NAMES_VI: dict[int, str] = {
    0: "Bạc lá lúa",
    1: "Đốm nâu",
}

# Các định dạng ảnh hợp lệ được xử lý trong pipeline
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Các phân tập dữ liệu chuẩn
SPLITS: tuple[str, str, str] = ("train", "val", "test")

# Tỷ lệ phân chia tập dữ liệu chuẩn (70% train, 15% val, 15% test)
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

# Danh sách tên các file ZIP nén dữ liệu nguồn mặc định
DEFAULT_ARCHIVES: tuple[Path, Path] = (
    Path("RiceLeafAnnotatedDataset.zip"),
    Path("dataset1.zip"),
)

