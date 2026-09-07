"""Khởi tạo và lưu mô hình dùng chung cho dịch vụ API."""

from functools import cache

from rice_leaf_detection.inference import DetectionPolicy, RiceLeafDetector

from .settings import get_settings


@cache
def get_detector() -> RiceLeafDetector:
    """Nạp mô hình singleton một lần duy nhất theo cấu hình runtime."""
    settings = get_settings()
    return RiceLeafDetector(
        weights=settings.weights,
        image_size=settings.image_size,
        confidence=settings.candidate_confidence,
        iou=settings.iou,
        policy=DetectionPolicy(
            review_threshold=settings.review_threshold,
            accept_threshold=settings.accept_threshold,
        ),
    )
