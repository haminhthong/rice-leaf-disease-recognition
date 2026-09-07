"""Kiểm tra metric theo ảnh và false-alarm benchmark."""

from rice_leaf_detection.constants import OUT_OF_SCOPE_NEGATIVE, TRUE_NEGATIVE
from rice_leaf_detection.evaluate import calculate_image_level_metrics


def test_image_level_metrics_tach_target_recall_va_negative_types() -> None:
    result = calculate_image_level_metrics(
        [
            {
                "truth_classes": [0],
                "detected_classes": [0],
                "annotation_status": "TARGET_POSITIVE",
                "accepted_detection": True,
            },
            {
                "truth_classes": [1],
                "detected_classes": [],
                "annotation_status": "TARGET_POSITIVE",
                "accepted_detection": False,
            },
            {
                "truth_classes": [],
                "detected_classes": [],
                "annotation_status": TRUE_NEGATIVE,
                "accepted_detection": False,
            },
            {
                "truth_classes": [],
                "detected_classes": [],
                "annotation_status": OUT_OF_SCOPE_NEGATIVE,
                "accepted_detection": True,
            },
        ]
    )

    assert result["image_target_recall"]["0"]["recall"] == 1.0
    assert result["image_target_recall"]["1"]["recall"] == 0.0
    assert result["negative_benchmark"][TRUE_NEGATIVE]["false_alarm_rate"] == 0.0
    assert result["negative_benchmark"][OUT_OF_SCOPE_NEGATIVE]["false_alarm_rate"] == 1.0
