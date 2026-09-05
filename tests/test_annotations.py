from pathlib import Path

import pytest

from rice_leaf_detection.annotations import normalize_class_name, parse_annotation_line


def test_taxonomy_alias_mapping() -> None:
    """Kiểm tra việc chuẩn hóa tên alias từ các nguồn dữ liệu về đúng 2 lớp bệnh chuẩn."""
    assert normalize_class_name("Bacterial Leaf Blight") == "Bacterial_Leaf_Blight"
    assert normalize_class_name("bacterial leafblight") == "Bacterial_Leaf_Blight"
    assert normalize_class_name("Brown-Spot") == "Brown_Spot"
    assert normalize_class_name("brown spot") == "Brown_Spot"
    assert normalize_class_name("unknown_disease") is None


def test_polygon_to_bbox_bounds() -> None:
    """Kiểm tra việc chuyển đổi tọa độ polygon sang bounding box nằm đúng trong giới hạn [0, 1]."""
    annotation, error = parse_annotation_line(
        "3 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8", {3: 1}, Path("nhan.txt"), 1
    )
    assert error is None
    assert annotation is not None
    assert annotation["class_id"] == 1
    assert annotation["x"] == pytest.approx(0.3)
    assert annotation["h"] == pytest.approx(0.6)
    assert 0.0 <= annotation["x"] <= 1.0
    assert 0.0 <= annotation["y"] <= 1.0
    assert 0.0 <= annotation["w"] <= 1.0
    assert 0.0 <= annotation["h"] <= 1.0


def test_invalid_annotation_rejected() -> None:
    """Kiểm tra từ chối các annotation có tọa độ ngoài miền, NaN hoặc chuỗi không hợp lệ."""
    # Tọa độ vượt quá [0, 1]
    ann1, err1 = parse_annotation_line("0 0.5 1.5 0.2 0.2", {0: 0}, Path("nhan.txt"), 1)
    assert ann1 is None
    assert err1 is not None

    # Tọa độ không phải số
    ann2, err2 = parse_annotation_line("0 0.5 abc 0.2 0.2", {0: 0}, Path("nhan.txt"), 2)
    assert ann2 is None
    assert err2 is not None

    # Tọa độ rỗng hoặc polygon dưới 3 điểm
    ann3, err3 = parse_annotation_line("0 0.1 0.2 0.3", {0: 0}, Path("nhan.txt"), 3)
    assert ann3 is None
    assert err3 is not None
