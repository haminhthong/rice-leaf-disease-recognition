"""Lớp suy luận dùng chung cho API và giao diện web.

Module đóng gói logic suy luận YOLOv8 và chuẩn hóa kết quả cho FastAPI và Streamlit.
CLI dùng trực tiếp chế độ truyền luồng của Ultralytics để xử lý ảnh, thư mục và video.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .constants import CLASS_NAMES_VI


@dataclass(frozen=True)
class Detection:
    """Thông tin chi tiết của một đối tượng tổn thương lá lúa được phát hiện."""

    class_id: int
    class_name: str
    class_name_vi: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]
    detection_score: float = 0.0

    def __post_init__(self) -> None:
        if self.detection_score == 0.0 and self.confidence != 0.0:
            object.__setattr__(self, "detection_score", self.confidence)


@dataclass(frozen=True)
class ImageSummary:
    """Tóm tắt phân tích mức ảnh phục vụ hỗ trợ trinh sát đồng ruộng (Decision Support)."""

    bacterial_leaf_blight_detected: bool = False
    brown_spot_detected: bool = False
    total_detections: int = 0
    requires_human_review: bool = False
    review_reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Prediction:
    """Kết quả dự đoán tổng hợp cho một hình ảnh."""

    detections: list[Detection]
    status: Literal["detected", "no_detection"]
    message: str
    warnings: list[str] = field(default_factory=list)
    image_summary: ImageSummary | None = None


class RiceLeafDetector:
    """Lớp suy luận chính cho phát hiện bệnh lá lúa.

    Args:
        weights: Đường dẫn tới file trọng số mô hình (`.pt` hoặc `.onnx`).
        image_size: Kích thước ảnh suy luận (mặc định 640).
        confidence: Ngưỡng tin cậy tối thiểu (mặc định 0.25).
        iou: Ngưỡng NMS IoU ghép hộp trùng (mặc định 0.45).
    """

    def __init__(
        self,
        weights: Path | str,
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
    ):
        if image_size <= 0:
            raise ValueError("Kích thước ảnh phải lớn hơn 0")
        if not 0 <= confidence <= 1:
            raise ValueError("Ngưỡng tin cậy phải nằm trong khoảng [0, 1]")
        if not 0 <= iou <= 1:
            raise ValueError("Ngưỡng IoU phải nằm trong khoảng [0, 1]")
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Không tìm thấy trọng số: {weights_path}")

        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou

    def predict(
        self,
        source: object,
        confidence: float | None = None,
        iou: float | None = None,
    ) -> tuple[Prediction, Any]:
        """Thực hiện phát hiện bệnh trên nguồn ảnh đầu vào (Path, numpy ndarray hoặc PIL Image).

        Args:
            source: Nguồn ảnh đầu vào.
            confidence: Ngưỡng tin cậy riêng cho lượt gọi hiện tại.
            iou: Ngưỡng IoU riêng cho lượt gọi hiện tại.

        Returns:
            tuple[Prediction, Any]:
                - Object `Prediction` đã được cấu trúc hóa kèm `ImageSummary`.
                - Đối tượng `Results` gốc của Ultralytics (phục vụ vẽ bounding box `.plot()`).
        """
        confidence = self.confidence if confidence is None else confidence
        iou = self.iou if iou is None else iou
        if not 0 <= confidence <= 1:
            raise ValueError("Ngưỡng tin cậy phải nằm trong khoảng [0, 1]")
        if not 0 <= iou <= 1:
            raise ValueError("Ngưỡng IoU phải nằm trong khoảng [0, 1]")

        result = self.model.predict(
            source=source,
            imgsz=self.image_size,
            conf=confidence,
            iou=iou,
            verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = str(self.model.names[class_id])
            coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
            conf_val = float(box.conf[0])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=name,
                    class_name_vi=CLASS_NAMES_VI.get(class_id, name),
                    confidence=conf_val,
                    detection_score=conf_val,
                    box_xyxy=coordinates,
                )
            )

        # Quyết định Image Summary và chính sách thẩm định của con người (Human Review Policy)
        has_blb = any(d.class_id == 0 for d in detections)
        has_brown_spot = any(d.class_id == 1 for d in detections)
        review_reasons: list[str] = []

        # 1. Phát hiện vùng có score ranh giới (borderline detection score)
        low_score_count = sum(1 for d in detections if 0.25 <= d.confidence < 0.45)
        if low_score_count > 0:
            review_reasons.append(
                f"Có {low_score_count} vùng tổn thương với điểm phát hiện ranh giới (0.25 - 0.45) "
                "cần chuyên gia kiểm tra trực quan."
            )

        # 2. Phát hiện các hộp chồng lấn khác lớp bệnh
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if detections[i].class_id != detections[j].class_id:
                    bi, bj = detections[i].box_xyxy, detections[j].box_xyxy
                    x1, y1 = max(bi[0], bj[0]), max(bi[1], bj[1])
                    x2, y2 = min(bi[2], bj[2]), min(bi[3], bj[3])
                    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    area_i = max(0.0, bi[2] - bi[0]) * max(0.0, bi[3] - bi[1])
                    area_j = max(0.0, bj[2] - bj[0]) * max(0.0, bj[3] - bj[1])
                    union = area_i + area_j - inter
                    if union > 0 and (inter / union) >= 0.30:
                        review_reasons.append(
                            "Phát hiện các hộp tổn thương chồng lấn thuộc hai lớp bệnh khác nhau, "
                            "nghi ngờ nhầm lẫn triệu chứng."
                        )
                        break

        image_summary = ImageSummary(
            bacterial_leaf_blight_detected=has_blb,
            brown_spot_detected=has_brown_spot,
            total_detections=len(detections),
            requires_human_review=len(review_reasons) > 0,
            review_reasons=review_reasons,
        )

        domain_warnings = [
            "Hệ thống là công cụ hỗ trợ trinh sát đồng ruộng (Field Scouting Support).",
            "TUYỆT ĐỐI KHÔNG tự động phun thuốc/hóa chất khi chưa có chỉ dẫn từ kỹ sư nông nghiệp.",
        ]

        if detections:
            status = "detected"
            message = f"Phát hiện {len(detections)} vùng tổn thương thuộc phạm vi mô hình hỗ trợ."
            warnings = domain_warnings
        else:
            status = "no_detection"
            message = "Không phát hiện vùng tổn thương nào vượt ngưỡng tin cậy thuộc 2 lớp hỗ trợ."
            warnings = [
                "Kết quả không khẳng định lá cây khỏe mạnh.",
                "Ảnh có thể thuộc bệnh ngoài phạm vi hỗ trợ hoặc ánh sáng không đạt chuẩn.",
                *domain_warnings,
            ]

        prediction = Prediction(
            detections=detections,
            status=status,
            message=message,
            warnings=warnings,
            image_summary=image_summary,
        )
        return prediction, result
