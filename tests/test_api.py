from fastapi.testclient import TestClient

from app.api import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tu_choi_dinh_dang_khong_ho_tro() -> None:
    response = TestClient(app).post(
        "/predict", files={"file": ("du-lieu.txt", b"abc", "text/plain")}
    )
    assert response.status_code == 415

