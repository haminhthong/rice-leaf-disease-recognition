import io
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from PIL import Image

from app.api import app
from app.settings import get_settings
from rice_leaf_detection.inference import Detection, Prediction


def test_health_live() -> None:
    response = TestClient(app).get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "live"}


@patch("app.api.get_detector", side_effect=FileNotFoundError("weights.pt"))
def test_health_ready_khi_thieu_weights(_mock_get_detector: MagicMock) -> None:
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Dịch vụ mô hình chưa sẵn sàng suy luận"


def test_info_dung_cau_hinh_runtime_mac_dinh() -> None:
    response = TestClient(app).get("/info")
    assert response.status_code == 200
    payload = response.json()
    settings = get_settings()
    assert payload["model_weights"] == settings.weights.as_posix()
    assert payload["max_upload_mb"] == 10


def test_tu_choi_dinh_dang_file_khong_ho_tro() -> None:
    response = TestClient(app).post(
        "/predict", files={"file": ("du-lieu.txt", b"abc_not_image_content", "text/plain")}
    )
    assert response.status_code == 415


def test_tu_choi_file_vuot_10_mb() -> None:
    large_content = b"a" * (10 * 1024 * 1024 + 1)
    response = TestClient(app).post(
        "/predict", files={"file": ("large.jpg", large_content, "image/jpeg")}
    )
    assert response.status_code == 413


@patch("app.api.get_detector")
def test_predict_thanh_cong_voi_model_mock(mock_get_detector: MagicMock) -> None:
    mock_detector = MagicMock()
    mock_prediction = Prediction(
        detections=[
            Detection(
                class_id=0,
                class_name="Bacterial_Leaf_Blight",
                class_name_vi="Bạc lá lúa",
                confidence=0.92,
                box_xyxy=(10.0, 20.0, 100.0, 200.0),
            )
        ],
        status="detected",
        message="Phát hiện 1 vùng bệnh",
        warnings=[],
    )
    mock_detector.predict.return_value = (mock_prediction, MagicMock())
    mock_get_detector.return_value = mock_detector

    # Tạo ảnh JPEG hợp lệ để chỉ kiểm tra hợp đồng API, không tải mô hình thật.
    img = Image.new("RGB", (100, 100), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")

    response = TestClient(app).post(
        "/predict", files={"file": ("sample.jpg", buffer.getvalue(), "image/jpeg")}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "detected"
    assert len(payload["detections"]) == 1
    assert payload["detections"][0]["class_name_vi"] == "Bạc lá lúa"


@patch("app.api.get_detector")
def test_predict_no_detection_voi_model_mock(mock_get_detector: MagicMock) -> None:
    mock_detector = MagicMock()
    mock_prediction = Prediction(
        detections=[],
        status="no_detection",
        message="Không phát hiện vùng bệnh",
        warnings=["Kết quả không khẳng định lá khỏe"],
    )
    mock_detector.predict.return_value = (mock_prediction, MagicMock())
    mock_get_detector.return_value = mock_detector

    img = Image.new("RGB", (100, 100), color="green")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")

    response = TestClient(app).post(
        "/predict", files={"file": ("clean_leaf.jpg", buffer.getvalue(), "image/jpeg")}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "no_detection"
    assert len(payload["detections"]) == 0
    assert len(payload["warnings"]) > 0
