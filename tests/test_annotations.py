from pathlib import Path

import pytest

from rice_leaf_detection.annotations import normalize_class_name, parse_annotation_line


def test_chuan_hoa_ten_lop() -> None:
    assert normalize_class_name("Bacterial Leaf Blight") == "Bacterial_Leaf_Blight"
    assert normalize_class_name("Brown-Spot") == "Brown_Spot"


def test_chuyen_polygon_thanh_bbox() -> None:
    annotation, error = parse_annotation_line(
        "3 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8", {3: 1}, Path("nhan.txt"), 1
    )
    assert error is None
    assert annotation is not None
    assert annotation["class_id"] == 1
    assert annotation["x"] == pytest.approx(0.3)
    assert annotation["h"] == pytest.approx(0.6)


def test_tu_choi_toa_do_ngoai_mien() -> None:
    annotation, error = parse_annotation_line(
        "0 0.5 1.2 0.2 0.2", {0: 0}, Path("nhan.txt"), 2
    )
    assert annotation is None
    assert error is not None

