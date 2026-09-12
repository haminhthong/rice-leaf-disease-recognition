"""Quản lý cấu hình runtime cho dịch vụ API và Dashboard."""

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class ApiSettings:
    """Cấu hình API được nạp từ biến môi trường."""

    weights: Path
    image_size: int
    confidence: float
    iou: float
    max_upload_bytes: int
    max_image_pixels: int
    cors_origins: tuple[str, ...]

    def validate(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"RICE_CONFIDENCE ({self.confidence}) phải thuộc [0, 1]")
        if not 0 <= self.iou <= 1:
            raise ValueError(f"RICE_IOU ({self.iou}) phải thuộc [0, 1]")
        if self.image_size <= 0:
            raise ValueError(f"RICE_IMAGE_SIZE ({self.image_size}) phải lớn hơn 0")
        if not self.cors_origins:
            raise ValueError("RICE_CORS_ORIGINS phải chứa ít nhất một nguồn được phép")


@cache
def get_settings() -> ApiSettings:
    """Tạo cấu hình từ biến môi trường."""
    default_weights = os.getenv("RICE_MODEL_PATH", "artifacts/model.pt")
    weights = Path(default_weights)
    settings = ApiSettings(
        weights=weights,
        image_size=int(os.getenv("RICE_IMAGE_SIZE", "640")),
        confidence=float(os.getenv("RICE_CONFIDENCE", "0.45")),
        iou=float(os.getenv("RICE_IOU", "0.45")),
        max_upload_bytes=10 * 1024 * 1024,
        max_image_pixels=25_000_000,
        cors_origins=tuple(
            origin.strip()
            for origin in os.getenv("RICE_CORS_ORIGINS", "http://localhost:8501").split(",")
            if origin.strip()
        ),
    )
    settings.validate()
    return settings
