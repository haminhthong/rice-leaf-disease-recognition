import pytest

from app.settings import get_settings


def test_doc_danh_sach_cors_tu_bien_moi_truong(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "RICE_CORS_ORIGINS",
        "https://demo.example, https://admin.example",
    )
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.cors_origins == (
            "https://demo.example",
            "https://admin.example",
        )
    finally:
        get_settings.cache_clear()


def test_tu_choi_cors_rong(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RICE_CORS_ORIGINS", " , ")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="RICE_CORS_ORIGINS"):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_doc_cau_hinh_confidence_tu_bien_moi_truong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RICE_CONFIDENCE", "0.55")
    monkeypatch.setenv("RICE_IOU", "0.50")
    monkeypatch.setenv("RICE_MODEL_PATH", "custom/best.pt")
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.confidence == 0.55
        assert settings.iou == 0.50
        assert settings.weights.as_posix() == "custom/best.pt"
    finally:
        get_settings.cache_clear()
