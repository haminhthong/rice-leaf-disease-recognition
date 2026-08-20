import hashlib
import json
import random
import sys
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


def configure_utf8_console() -> None:
    """Dùng UTF-8 cho thông báo dòng lệnh trên Windows và các terminal cũ."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@lru_cache(maxsize=None)
def _dct_matrix(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float64)
    frequencies = positions[:, None]
    matrix = np.cos(np.pi * (positions + 0.5) * frequencies / size)
    matrix[0] *= np.sqrt(1 / size)
    matrix[1:] *= np.sqrt(2 / size)
    return matrix


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Tính pHash 64 bit từ ảnh bằng biến đổi cosin rời rạc."""
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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract_zip(archive: Path, destination: Path) -> None:
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
