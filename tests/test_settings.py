from pathlib import Path

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


def test_doc_policy_da_tune_canh_model_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    policy_path = tmp_path / "detection_policy.json"
    policy_path.write_text(
        '{"review_threshold": 0.31, "accept_threshold": 0.67}',
        encoding="utf-8",
    )
    monkeypatch.setenv("RICE_MODEL_PATH", str(tmp_path / "model.pt"))
    monkeypatch.delenv("RICE_REVIEW_THRESHOLD", raising=False)
    monkeypatch.delenv("RICE_ACCEPT_THRESHOLD", raising=False)
    get_settings.cache_clear()
    try:
        settings = get_settings()
        assert settings.review_threshold == 0.31
        assert settings.accept_threshold == 0.67
    finally:
        get_settings.cache_clear()
