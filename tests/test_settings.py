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
