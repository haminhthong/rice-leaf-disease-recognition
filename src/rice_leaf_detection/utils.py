"""Các hàm tiện ích hệ thống dùng chung cho dự án Rice Leaf Disease Recognition.

Module này cung cấp các tiện ích xử lý IO, mã hóa hash (pHash 64-bit qua DCT, SHA-256),
cố định seed ngẫu nhiên cho tính tái lập và giải nén zip an toàn phòng chống lỗ hổng Path Traversal.
"""

import hashlib
import json
import random
import sys
import zipfile
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image


def configure_utf8_console() -> None:
    """Cấu hình bộ mã hóa UTF-8 cho stdout/stderr trên Windows PowerShell và CMD."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def seed_everything(seed: int) -> None:
    """Cố định seed ngẫu nhiên cho Python, NumPy và PyTorch để đảm bảo kết quả tái lập được."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@cache
def _dct_matrix(size: int) -> np.ndarray:
    """Tạo ma trận biến đổi Cosin rời rạc (Discrete Cosine Transform - DCT-II).

    Ma trận được lưu trong bộ nhớ đệm (cache) để tránh tính toán lại nhiều lần khi hash ảnh.
    """
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (positions + 0.5) * frequencies / size)
    matrix[0] *= np.sqrt(1 / size)
    matrix[1:] *= np.sqrt(2 / size)
    return matrix


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Tính chuỗi perceptual hash (pHash) 64-bit từ ảnh bằng biến đổi DCT.

    Thuật toán:
    1. Chuyển ảnh sang ảnh xám (Grayscale).
    2. Resize ảnh về kích thước (4 * hash_size, 4 * hash_size) bằng bộ lọc LANCZOS.
    3. Áp dụng biến đổi DCT 2D để trích xuất tần số không gian.
    4. Lấy ma trận tần số thấp (kích thước hash_size x hash_size).
    5. So sánh từng phần tử với giá trị trung vị (median) để tạo chuỗi 64 bit nhị phân.
    6. Chuyển chuỗi bit thành mã Hexadecimal.

    Args:
        image: Ảnh PIL cần tính hash.
        hash_size: Kích thước lưới tần số (mặc định 8x8 = 64 bit).

    Returns:
        str: Chuỗi hex đại diện cho pHash của ảnh.
    """
    if hash_size <= 0:
        raise ValueError("Kích thước pHash phải lớn hơn 0")
    dct_size = hash_size * 4
    grayscale = image.convert("L").resize((dct_size, dct_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(grayscale, dtype=np.float64)
    transform = _dct_matrix(dct_size)
    coefficients = transform @ pixels @ transform.T
    low_frequencies = coefficients[:hash_size, :hash_size]
    bits = low_frequencies > np.median(low_frequencies)
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Tính mã băm SHA-256 của một file theo từng block bộ nhớ để tiết kiệm RAM.

    Args:
        path: Đường dẫn tới file cần tính hash.
        chunk_size: Dung lượng mỗi khối đọc (mặc định 1MB).

    Returns:
        str: Chuỗi hex checksum SHA-256.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(archive: Path, destination: Path) -> None:
    """Giải nén file ZIP an toàn, phòng chống tấn công chèn đường dẫn (Zip Slip / Path Traversal).

    Args:
        archive: Đường dẫn file ZIP đầu vào.
        destination: Thư mục đích giải nén.

    Raises:
        FileNotFoundError: Nếu file ZIP không tồn tại.
        ValueError: Nếu file ZIP không hợp lệ hoặc chứa đường dẫn thoát ra khỏi thư mục đích.
    """
    if not archive.exists():
        raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {archive}")
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Không phải ZIP hợp lệ: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"ZIP chứa đường dẫn không an toàn: {member.filename}")
        zip_file.extractall(destination)


def write_json(path: Path, value: object) -> None:
    """Ghi cấu trúc dữ liệu Python ra file JSON với bộ mã UTF-8 và định dạng thụt lề 2 spaces."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
