"""Quản lý cấu hình khi vận hành dịch vụ API."""

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class ApiSettings:
    """Cấu hình API được tải từ biến môi trường và kiểm tra ràng buộc."""

    weights: Path
    image_size: int
    candidate_confidence: float
    iou: float
    review_threshold: float
    accept_threshold: float
    max_upload_bytes: int
    max_image_pixels: int
    inference_concurrency: int
    cors_origins: tuple[str, ...]

    @property
    def confidence(self) -> float:
        """Alias đọc tương thích với cấu hình runtime cũ."""
        return self.candidate_confidence

    def validate(self) -> None:
        """Kiểm tra ràng buộc giá trị hợp lệ của cấu hình."""
        if not 0 <= self.candidate_confidence <= 1:
            raise ValueError(
                f"RICE_CANDIDATE_CONFIDENCE ({self.candidate_confidence}) phải thuộc [0, 1]"
            )
        if not 0 <= self.iou <= 1:
            raise ValueError(f"RICE_IOU ({self.iou}) phải thuộc [0, 1]")
        if not 0 <= self.review_threshold <= 1:
            raise ValueError(f"RICE_REVIEW_THRESHOLD ({self.review_threshold}) phải thuộc [0, 1]")
        if not 0 <= self.accept_threshold <= 1:
            raise ValueError(f"RICE_ACCEPT_THRESHOLD ({self.accept_threshold}) phải thuộc [0, 1]")
        if self.review_threshold > self.accept_threshold:
            raise ValueError("RICE_REVIEW_THRESHOLD không được lớn hơn RICE_ACCEPT_THRESHOLD")
        if self.image_size <= 0:
            raise ValueError(f"RICE_IMAGE_SIZE ({self.image_size}) phải lớn hơn 0")
        if self.inference_concurrency <= 0:
            raise ValueError(f"INFERENCE_CONCURRENCY ({self.inference_concurrency}) phải lớn hơn 0")
        if not self.cors_origins:
            raise ValueError("RICE_CORS_ORIGINS phải chứa ít nhất một nguồn được phép")


@cache
def get_settings() -> ApiSettings:
    """Tạo cấu hình từ biến môi trường và kiểm tra các ràng buộc."""
    release_dir = os.getenv("RICE_MODEL_RELEASE_DIR")
    default_weights = str(Path(release_dir) / "model.pt") if release_dir else "artifacts/model.pt"
    settings = ApiSettings(
        weights=Path(os.getenv("RICE_MODEL_PATH", default_weights)),
        image_size=int(os.getenv("RICE_IMAGE_SIZE", "640")),
        candidate_confidence=float(
            os.getenv("RICE_CANDIDATE_CONFIDENCE", os.getenv("RICE_CONFIDENCE", "0.20"))
        ),
        iou=float(os.getenv("RICE_IOU", "0.45")),
        review_threshold=float(os.getenv("RICE_REVIEW_THRESHOLD", "0.20")),
        accept_threshold=float(os.getenv("RICE_ACCEPT_THRESHOLD", "0.45")),
        max_upload_bytes=10 * 1024 * 1024,
        max_image_pixels=25_000_000,
        inference_concurrency=int(os.getenv("INFERENCE_CONCURRENCY", "2")),
        cors_origins=tuple(
            origin.strip()
            for origin in os.getenv("RICE_CORS_ORIGINS", "http://localhost:8501").split(",")
            if origin.strip()
        ),
    )
    settings.validate()
    return settings
