"""Unit tests cho module inference.py (RiceLeafDetector, Detection, Prediction)."""

from pathlib import Path

import pytest

from rice_leaf_detection.inference import (
    Detection,
    DetectionPolicy,
    Prediction,
    RawDetection,
    RiceLeafDetector,
)


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


def test_prediction_dataclass_fields() -> None:
    pred = Prediction(
        detections=[],
        status="no_detection",
        message="Không phát hiện vùng tổn thương",
        warnings=["Kết quả không khẳng định lá khỏe"],
    )
    assert pred.status == "no_detection"
    assert len(pred.warnings) == 1


def test_detection_policy_phan_biet_accept_review_discard() -> None:
    policy = DetectionPolicy(review_threshold=0.20, accept_threshold=0.45)
    candidates = [
        RawDetection(0, "Bacterial_Leaf_Blight", 0.10, (0, 0, 1, 1)),
        RawDetection(0, "Bacterial_Leaf_Blight", 0.30, (0, 0, 1, 1)),
        RawDetection(1, "Brown_Spot", 0.80, (0, 0, 1, 1)),
    ]

    detections = policy.apply(candidates)

    assert [d.decision for d in detections] == ["review", "accepted"]
    assert all(not hasattr(d, "detection_score") for d in detections)
    assert [d.score for d in detections] == [0.30, 0.80]


def test_detector_tu_choi_confidence_iou_ngoai_mien() -> None:
    with pytest.raises(ValueError, match="Ngưỡng tin cậy"):
        RiceLeafDetector("weights.pt", confidence=-0.1)

    with pytest.raises(ValueError, match="Ngưỡng IoU"):
        RiceLeafDetector("weights.pt", confidence=0.5, iou=1.5)

    with pytest.raises(ValueError, match="Kích thước ảnh"):
        RiceLeafDetector("weights.pt", image_size=0)


def test_detector_tu_choi_file_weights_khong_ton_tai() -> None:
    with pytest.raises(FileNotFoundError, match="Không tìm thấy trọng số"):
        RiceLeafDetector(Path("file_trong_so_khong_ton_tai.pt"))
