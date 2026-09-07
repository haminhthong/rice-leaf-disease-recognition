"""Khai báo cấu trúc phản hồi Pydantic dùng trong tài liệu OpenAPI."""

from typing import Literal

from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    """Thông tin chi tiết một hộp tổn thương được phát hiện."""

    class_id: int = Field(description="ID lớp bệnh (0: Bạc lá, 1: Đốm nâu)")
    class_name: str = Field(description="Tên lớp tiếng Anh gốc")
    class_name_vi: str = Field(description="Tên lớp tiếng Việt")
    score: float = Field(description="Detection score của mô hình [0, 1], không phải xác suất bệnh")
    decision: Literal["accepted", "review"] = Field(
        description="Quyết định của policy: chấp nhận hoặc cần người kiểm tra"
    )
    box_xyxy: tuple[float, float, float, float] = Field(
        description="Tọa độ Bounding Box (xmin, ymin, xmax, ymax)"
    )


class ImageSummaryResponse(BaseModel):
    """Tóm tắt phân tích mức ảnh phục vụ trinh sát đồng ruộng (Field Scouting Decision Support)."""

    bacterial_leaf_blight_detected: bool = Field(description="Có tổn thương Bạc lá lúa hay không")
    brown_spot_detected: bool = Field(description="Có tổn thương Đốm nâu hay không")
    total_detections: int = Field(description="Tổng số vùng tổn thương phát hiện được")
    requires_human_review: bool = Field(
        description="Cờ yêu cầu chuyên gia/kỹ sư nông nghiệp thẩm định"
    )
    review_reasons: list[str] = Field(
        default_factory=list, description="Lý do cần chuyên gia thẩm định"
    )
    accepted_detections: int = Field(default=0, description="Số box được chấp nhận")
    review_detections: int = Field(default=0, description="Số box cần người kiểm tra")


class PredictionResponse(BaseModel):
    """Cấu trúc phản hồi tổng hợp cho yêu cầu phát hiện bệnh."""

    filename: str | None = Field(default=None, description="Tên file ảnh upload")
    status: Literal["DETECTED", "REVIEW_REQUIRED", "NO_SUPPORTED_SYMPTOM_DETECTED"] = Field(
        description="Trạng thái quyết định của detector"
    )
    message: str = Field(description="Thông điệp mô tả kết quả")
    image_summary: ImageSummaryResponse | None = Field(
        default=None, description="Tóm tắt phân tích mức ảnh phục vụ hỗ trợ quyết định"
    )
    image_quality: dict = Field(
        default_factory=dict, description="Kết quả quality gate blur/phơi sáng/kích thước"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Cảnh báo giới hạn phạm vi mô hình và miễn trừ trách nhiệm",
    )
    detections: list[DetectionResponse] = Field(
        default_factory=list, description="Danh sách các hộp tổn thương phát hiện được"
    )
