"""Unit tests cho module inference.py (RiceLeafDetector, Detection, Prediction)."""

from pathlib import Path

import pytest

from rice_leaf_detection.inference import Detection, Prediction, RiceLeafDetector


def test_detection_dataclass_creation() -> None:
    det = Detection(
        class_id=0,
        class_name="Bacterial_Leaf_Blight",
        class_name_vi="Bạc lá lúa",
        confidence=0.85,
        box_xyxy=(10.0, 20.0, 100.0, 200.0),
    )
    assert det.class_id == 0
    assert det.class_name_vi == "Bạc lá lúa"
    assert det.confidence == 0.85


def test_prediction_dataclass_rejection() -> None:
    pred = Prediction(detections=[], rejected=True, reason="Không có vùng bệnh")
    assert pred.rejected is True
    assert pred.reason == "Không có vùng bệnh"


def test_detector_tu_choi_confidence_iou_ngoai_mien() -> None:
    with pytest.raises(ValueError, match="Ngưỡng tin cậy"):
        RiceLeafDetector("weights.pt", confidence=-0.1)

    with pytest.raises(ValueError, match="Ngưỡng IoU"):
        RiceLeafDetector("weights.pt", confidence=0.5, iou=1.5)


def test_detector_tu_choi_file_weights_khong_ton_tai() -> None:
    with pytest.raises(FileNotFoundError, match="Không tìm thấy trọng số"):
        RiceLeafDetector(Path("file_trong_so_khong_ton_tai.pt"))
