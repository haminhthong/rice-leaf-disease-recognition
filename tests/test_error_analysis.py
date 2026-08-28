import pytest

from rice_leaf_detection.error_analysis import box_iou, match_detections, parse_boolean


def test_iou_cua_hai_box_giong_nhau() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_ghep_prediction_tao_tp_fp_fn() -> None:
    truth = [(0, (0.0, 0.0, 10.0, 10.0), 1.0), (1, (20.0, 20.0, 30.0, 30.0), 1.0)]
    predictions = [(0, (0.0, 0.0, 10.0, 10.0), 0.9), (0, (40.0, 40.0, 50.0, 50.0), 0.8)]
    errors, counts = match_detections(truth, predictions)
    assert counts["true_positive"] == 1
    assert counts["false_positive"] == 1
    assert counts["false_negative"] == 1
    assert len(errors) == 2


def test_chuan_hoa_boolean_tu_csv() -> None:
    assert parse_boolean("True") is True
    assert parse_boolean("False") is False
