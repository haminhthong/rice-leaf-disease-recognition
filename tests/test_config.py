from pathlib import Path

import pytest

from rice_leaf_detection.config import load_config


def test_doc_cau_hinh_mac_dinh() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config.project.seed == 42
    assert config.data.image_size == 640


def test_cau_hinh_thieu_nhom_bat_buoc(tmp_path: Path) -> None:
    path = tmp_path / "sai.yaml"
    path.write_text("project: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Thiếu"):
        load_config(path)


def test_tu_choi_gia_tri_cau_hinh_khong_hop_le(tmp_path: Path) -> None:
    content = Path("configs/default.yaml").read_text(encoding="utf-8")
    path = tmp_path / "sai.yaml"
    path.write_text(content.replace("image_size: 640", "image_size: 0"), encoding="utf-8")
    with pytest.raises(ValueError, match="data.image_size"):
        load_config(path)


def test_tu_choi_trong_so_mo_hinh_null(tmp_path: Path) -> None:
    content = Path("configs/default.yaml").read_text(encoding="utf-8")
    path = tmp_path / "sai.yaml"
    path.write_text(
        content.replace("weights: yolov8s.pt", "weights:"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model.weights"):
        load_config(path)
