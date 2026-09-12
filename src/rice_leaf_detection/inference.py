"""Module suy luận (Inference) và phát hiện tổn thương lá lúa bằng YOLOv8."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import CLASS_NAMES_VI


@dataclass(frozen=True)
class Detection:
    """Biểu diễn một vùng tổn thương được phát hiện trên phiến lá."""

    class_id: int
    class_name: str
    class_name_vi: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    """Kết quả dự đoán tổng thể cho một bức ảnh."""

    detections: list[Detection]
    status: str  # "DETECTED" hoặc "NO_SYMPTOM_DETECTED"
    message: str
    warnings: list[str] = field(default_factory=list)


class RiceLeafDetector:
    """Detector sử dụng YOLOv8 phát hiện tổn thương Bạc lá lúa và Đốm nâu."""

    def __init__(
        self,
        weights: Path | str,
        image_size: int = 640,
        confidence: float = 0.45,
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
        """Thực hiện phát hiện tổn thương trên ảnh đầu vào (numpy ndarray, PIL hoặc Path).

        Args:
            source: Nguồn ảnh đầu vào.
            confidence: Ngưỡng tin cậy (mặc định lấy theo detector).
            iou: Ngưỡng NMS IoU (mặc định lấy theo detector).

        Returns:
            tuple[Prediction, Any]:
                - Object Prediction chứa danh sách detections và thông điệp.
                - Object Results của Ultralytics để vẽ bounding box (.plot()).
        """
        conf_thresh = self.confidence if confidence is None else confidence
        iou_thresh = self.iou if iou is None else iou

        if not 0 <= conf_thresh <= 1:
            raise ValueError("Ngưỡng tin cậy phải nằm trong khoảng [0, 1]")
        if not 0 <= iou_thresh <= 1:
            raise ValueError("Ngưỡng IoU phải nằm trong khoảng [0, 1]")

        result = self.model.predict(
            source=source,
            imgsz=self.image_size,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )[0]

        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = str(self.model.names[class_id])
            conf_val = float(box.conf[0])
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=name,
                    class_name_vi=CLASS_NAMES_VI.get(class_id, name),
                    confidence=conf_val,
                    box_xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
                )
            )

        disclaimers = [
            (
                "Hệ thống hỗ trợ trinh sát và định vị tổn thương thực địa, "
                "không thay thế chẩn đoán chuyên môn."
            ),
            (
                "Không tự ý phun thuốc bảo vệ thực vật khi chưa có "
                "hướng dẫn từ chuyên gia nông nghiệp."
            ),
        ]

        if detections:
            status = "DETECTED"
            message = f"Phát hiện {len(detections)} vùng tổn thương đạt ngưỡng tin cậy."
            warnings = disclaimers
        else:
            status = "NO_SYMPTOM_DETECTED"
            message = "Không phát hiện triệu chứng bệnh thuộc phạm vi hỗ trợ ở ngưỡng đã chọn."
            warnings = [
                "Kết quả không khẳng định phiến lá khỏe mạnh hoàn toàn.",
                *disclaimers,
            ]

        prediction = Prediction(
            detections=detections,
            status=status,
            message=message,
            warnings=warnings,
        )
        return prediction, result
