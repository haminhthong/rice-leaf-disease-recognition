"""Kiểm tra và giải mã file ảnh tải lên an toàn (Upload Image Validation)."""

import cv2
import numpy as np
from fastapi import HTTPException

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}


def validate_magic_bytes(content: bytes) -> str:
    """Kiểm tra magic bytes thực sự của tập tin để phòng ngừa giả mạo Content-Type."""
    if len(content) < 12:
        raise HTTPException(status_code=400, detail="Tập tin quá nhỏ hoặc bị hỏng")

    # Kiểm tra chữ ký nhị phân thay vì tin vào Content-Type do máy khách gửi.
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp"

    raise HTTPException(
        status_code=415,
        detail="Định dạng tập tin không hỗ trợ. Chỉ chấp nhận ảnh JPEG, PNG hoặc WebP.",
    )


def decode_and_validate_image(content: bytes, max_pixels: int = 25_000_000) -> np.ndarray:
    """Giải mã ảnh bằng OpenCV và kiểm tra giới hạn điểm ảnh phòng chống Decompression Bomb."""
    validate_magic_bytes(content)

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Không thể giải mã nội dung ảnh")

    height, width = image.shape[:2]
    if width * height > max_pixels:
        total = width * height
        raise HTTPException(
            status_code=413,
            detail=f"Ảnh quá lớn ({width}x{height} = {total}px). Giới hạn {max_pixels}px.",
        )

    return image
