from pathlib import Path

import pytest

from rice_leaf_detection.config import load_config


def test_doc_cau_hinh_mac_dinh() -> None:
    config = load_config(Path("configs/default.yaml"))
    assert config["project"]["seed"] == 42
    assert config["data"]["image_size"] == 640


def test_cau_hinh_thieu_nhom_bat_buoc(tmp_path: Path) -> None:
    path = tmp_path / "sai.yaml"
    path.write_text("project: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="thiếu"):
        load_config(path)

