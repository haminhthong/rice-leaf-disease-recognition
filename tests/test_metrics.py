"""Kiểm tra tính toán các độ đo chuẩn trong Object Detection (Precision, Recall, IoU)."""

import pytest

from rice_leaf_detection.error_analysis import box_iou


def test_box_iou_perfect_overlap() -> None:
    box = (10.0, 10.0, 50.0, 50.0)
    assert box_iou(box, box) == pytest.approx(1.0)


def test_box_iou_disjoint_boxes() -> None:
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (20.0, 20.0, 30.0, 30.0)
    assert box_iou(box1, box2) == 0.0


def test_box_iou_partial_overlap() -> None:
    # box1: 0 to 10 on both axes, area = 100
    # box2: 5 to 15 on both axes, area = 100
    # intersection: 5 to 10 on both axes, area = 25
    # union: 100 + 100 - 25 = 175 -> IoU = 25 / 175 = 1/7
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 5.0, 15.0, 15.0)
    expected_iou = 25.0 / 175.0
    assert box_iou(box1, box2) == pytest.approx(expected_iou, rel=1e-3)


def test_precision_recall_formula() -> None:
    # Kiểm tra tính toán cơ bản: Precision = TP / (TP + FP), Recall = TP / (TP + FN)
    tp, fp, fn = 8, 2, 2
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    assert precision == 0.8
    assert recall == 0.8
