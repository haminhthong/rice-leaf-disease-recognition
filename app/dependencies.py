"""Khởi tạo và lưu mô hình dùng chung cho dịch vụ API."""

from functools import cache

from rice_leaf_detection.inference import RiceLeafDetector

from .settings import get_settings


@cache
def get_detector() -> RiceLeafDetector:
    """Nạp mô hình singleton một lần duy nhất theo cấu hình runtime."""
    settings = get_settings()
    return RiceLeafDetector(
        weights=settings.weights,
        image_size=settings.image_size,
        confidence=settings.confidence,
        iou=settings.iou,
    )
