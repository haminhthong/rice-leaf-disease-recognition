"""Khai báo cấu trúc phản hồi Pydantic dùng trong API và OpenAPI docs."""

from pydantic import BaseModel, Field


class DetectionResponse(BaseModel):
    """Thông tin chi tiết một hộp tổn thương được phát hiện."""

    class_id: int = Field(description="ID lớp bệnh (0: Bạc lá, 1: Đốm nâu)")
    class_name: str = Field(description="Tên lớp tiếng Anh gốc")
    class_name_vi: str = Field(description="Tên lớp tiếng Việt")
    confidence: float = Field(description="Độ tin cậy của mô hình [0, 1]")
    box_xyxy: tuple[float, float, float, float] = Field(
        description="Tọa độ Bounding Box (xmin, ymin, xmax, ymax)"
    )


class PredictionResponse(BaseModel):
    """Cấu trúc phản hồi cho yêu cầu phát hiện bệnh."""

    filename: str | None = Field(default=None, description="Tên file ảnh upload")
    status: str = Field(description="Trạng thái kết quả (DETECTED hoặc NO_SYMPTOM_DETECTED)")
    message: str = Field(description="Thông điệp mô tả kết quả")
    warnings: list[str] = Field(
        default_factory=list,
        description="Cảnh báo khuyến cáo chuyên môn nông nghiệp",
    )
    detections: list[DetectionResponse] = Field(
        default_factory=list, description="Danh sách các hộp tổn thương phát hiện được"
    )
