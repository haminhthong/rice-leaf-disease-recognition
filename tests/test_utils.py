"""Unit tests cho module utils.py (pHash, SHA-256, Zip extraction)."""

import zipfile
from pathlib import Path

import pytest
from PIL import Image

from rice_leaf_detection.utils import (
    configure_utf8_console,
    perceptual_hash,
    safe_extract_zip,
    seed_everything,
    sha256_file,
)


def test_perceptual_hash_tinh_hash_anh() -> None:
    img = Image.new("RGB", (100, 100), color="green")
    h1 = perceptual_hash(img)
    assert isinstance(h1, str)
    assert len(h1) == 16  # 64-bit hex string

    # Kiểm tra hai ảnh giống nhau có cùng hash
    h2 = perceptual_hash(img)
    assert h1 == h2


def test_sha256_file(tmp_path: Path) -> None:
    test_file = tmp_path / "test.txt"
    test_file.write_text("Rice Leaf Disease Recognition", encoding="utf-8")
    digest = sha256_file(test_file)
    assert len(digest) == 64  # SHA-256 hex string


def test_safe_extract_zip_tu_choi_zip_slip(tmp_path: Path) -> None:
    zip_path = tmp_path / "malicious.zip"
    dest_path = tmp_path / "extracted"

    # Đường dẫn lùi ra ngoài thư mục đích phải bị từ chối.
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../evil.txt", "malicious content")

    with pytest.raises(ValueError, match="đường dẫn không an toàn"):
        safe_extract_zip(zip_path, dest_path)


def test_configure_utf8_console_va_seed() -> None:
    configure_utf8_console()
    seed_everything(42)
