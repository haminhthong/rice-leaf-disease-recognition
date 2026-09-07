"""Kiểm tra và giải mã file ảnh tải lên an toàn (Upload Image Validation)."""

import io

import numpy as np
from fastapi import HTTPException
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover - fallback cho API tối giản
    cv2 = None

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

    if cv2 is not None:
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        # Fallback dùng Pillow khi OpenCV chưa có; kết quả vẫn trả BGR như cv2.
        try:
            with Image.open(io.BytesIO(content)) as decoded:
                image = np.asarray(decoded.convert("RGB"))[:, :, ::-1].copy()
        except Exception:
            image = None
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


def assess_image_quality(image: np.ndarray) -> dict[str, object]:
    """Đánh giá nhanh blur, phơi sáng và kích thước trước khi gọi model.

    Đây là quality gate hỗ trợ cảnh báo, không tự suy ra ảnh healthy. API vẫn
    cho model chạy để trả kết quả có cảnh báo thay vì biến lỗi chất lượng ảnh
    thành một kết luận sinh học.
    """
    height, width = image.shape[:2]
    if cv2 is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    else:
        gray = np.dot(image[..., :3], np.array([0.114, 0.587, 0.299]))
        gradient_y = np.diff(gray, axis=0)
        gradient_x = np.diff(gray, axis=1)
        blur_score = float(np.var(gradient_x) + np.var(gradient_y))
    mean_intensity = float(gray.mean())
    warnings: list[str] = []
    if min(width, height) < 64:
        warnings.append("Kích thước cạnh nhỏ; kết quả định vị có thể kém ổn định.")
    if blur_score < 20:
        warnings.append("Ảnh bị mờ mạnh; nên chụp lại ở khoảng cách và ánh sáng tốt hơn.")
    elif blur_score < 60:
        warnings.append("Ảnh hơi mờ; cần thận trọng khi đọc các tổn thương nhỏ.")
    if mean_intensity < 25:
        warnings.append("Ảnh quá tối; kết quả có thể bị ảnh hưởng bởi thiếu sáng.")
    elif mean_intensity > 235:
        warnings.append("Ảnh quá sáng; vùng tổn thương có thể bị cháy sáng.")
    return {
        "status": "review" if blur_score < 20 else "acceptable",
        "blur_score": round(blur_score, 3),
        "mean_intensity": round(mean_intensity, 3),
        "width": width,
        "height": height,
        "warnings": warnings,
    }
