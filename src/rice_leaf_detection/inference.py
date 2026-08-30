"""Lớp suy luận dùng chung (Unified Inference Module).

Module này đóng gói logic suy luận YOLOv8 để tái sử dụng thống nhất trên:
1. Giao diện dòng lệnh CLI (`rice-predict`).
2. RESTful Web API (`FastAPI`).
3. Giao diện người dùng Web Dashboard (`Streamlit`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import CLASS_NAMES_VI


@dataclass(frozen=True)
class Detection:
    """Thông tin chi tiết của một đối tượng tổn thương lá lúa được phát hiện."""

    class_id: int
    class_name: str
    class_name_vi: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    """Kết quả dự đoán tổng hợp cho một hình ảnh."""

    detections: list[Detection]
    rejected: bool
    reason: str | None


class RiceLeafDetector:
    """Lớp suy luận chính cho phát hiện bệnh lá lúa.

    Args:
        weights: Đường dẫn tới file trọng số mô hình (`.pt` hoặc `.onnx`).
        confidence: Ngưỡng tin cậy tối thiểu (mặc định 0.25).
        iou: Ngưỡng NMS IoU ghép hộp trùng (mặc định 0.45).
    """

    def __init__(self, weights: Path | str, confidence: float = 0.25, iou: float = 0.45):
        if not 0 <= confidence <= 1:
            raise ValueError("Ngưỡng tin cậy phải nằm trong khoảng [0, 1]")
        if not 0 <= iou <= 1:
            raise ValueError("Ngưỡng IoU phải nằm trong khoảng [0, 1]")
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"Không tìm thấy trọng số: {weights_path}")

        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.confidence = confidence
        self.iou = iou

    def predict(self, source: object) -> tuple[Prediction, Any]:
        """Thực hiện phát hiện bệnh trên nguồn ảnh đầu vào (Path, numpy ndarray hoặc PIL Image).

        Args:
            source: Nguồn ảnh đầu vào.

        Returns:
            tuple[Prediction, Any]:
                - Object `Prediction` đã được cấu trúc hóa.
                - Đối tượng `Results` gốc của Ultralytics (phục vụ vẽ bounding box `.plot()`).
        """
        result = self.model.predict(
            source=source,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )[0]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            name = str(self.model.names[class_id])
            coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=name,
                    class_name_vi=CLASS_NAMES_VI.get(class_id, name),
                    confidence=float(box.conf[0]),
                    box_xyxy=coordinates,
                )
            )
        prediction = Prediction(
            detections=detections,
            rejected=not detections,
            reason="Không có vùng bệnh đạt ngưỡng tin cậy" if not detections else None,
        )
        return prediction, result

