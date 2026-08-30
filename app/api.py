"""FastAPI Web Service cho hệ thống Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Recognition API).

Dịch vụ RESTful API cung cấp các endpoint:
- `GET /health`: Kiểm tra trạng thái hoạt động của dịch vụ (Health Check).
- `GET /info`: Thông tin cấu hình mô hình, các lớp bệnh được hỗ trợ và ngưỡng suy luận.
- `POST /predict`: Upload ảnh lá lúa (JPEG/PNG/WebP, <= 10MB) và nhận kết quả phát hiện dạng JSON.
"""

import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from rice_leaf_detection.constants import CLASS_NAMES, CLASS_NAMES_VI
from rice_leaf_detection.inference import RiceLeafDetector

app = FastAPI(
    title="Rice Leaf Disease Detection API",
    description=(
        "API phát hiện bạc lá lúa (Bacterial Leaf Blight) "
        "và đốm nâu (Brown Spot) bằng YOLOv8."
    ),

    version="1.1.0",
)

# Giới hạn dung lượng upload tối đa 10 MB để phòng chống tấn công Từ chối dịch vụ (DoS)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@lru_cache(maxsize=1)
def get_detector() -> RiceLeafDetector:
    """Khởi tạo và nạp mô hình vào bộ nhớ đệm (LRU cache) để tối ưu latency suy luận."""
    weights = Path(os.getenv("RICE_MODEL_PATH", "artifacts/best.pt"))
    confidence = float(os.getenv("RICE_CONFIDENCE", "0.25"))
    iou = float(os.getenv("RICE_IOU", "0.45"))
    return RiceLeafDetector(weights, confidence=confidence, iou=iou)


@app.get("/health", summary="Kiểm tra sức khỏe dịch vụ")
def health() -> dict[str, str]:
    """Endpoint Healthcheck phục vụ kiểm tra trạng thái hoạt động của Container/Server."""
    return {"status": "ok"}


@app.get("/info", summary="Thông tin cấu hình mô hình")
def info() -> dict:
    """Trả về danh sách các lớp bệnh được hỗ trợ, ngưỡng tin cậy và trọng số mô hình."""
    weights = os.getenv("RICE_MODEL_PATH", "artifacts/best.pt")
    confidence = float(os.getenv("RICE_CONFIDENCE", "0.25"))
    iou = float(os.getenv("RICE_IOU", "0.45"))
    return {
        "service": "Rice Leaf Disease Detection API",
        "version": "1.1.0",
        "supported_classes": CLASS_NAMES,
        "supported_classes_vi": CLASS_NAMES_VI,
        "model_weights": weights,
        "default_confidence": confidence,
        "default_iou": iou,
        "max_upload_mb": 10,
    }


@app.post("/predict", summary="Dự đoán bệnh trên ảnh lá lúa")
async def predict(file: Annotated[UploadFile, File()]) -> dict:
    """Endpoint tiếp nhận file ảnh upload và trả về danh sách bounding boxes phát hiện bệnh.

    - Kiểm tra MIME type hợp lệ (JPEG, PNG, WebP).
    - Giới hạn kích thước file <= 10MB.
    - Giải mã ảnh bằng OpenCV và chạy suy luận với `RiceLeafDetector`.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Định dạng file không được hỗ trợ. Chỉ chấp nhận JPEG, PNG hoặc WebP.",
        )
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Dung lượng ảnh vượt quá giới hạn 10 MB.",
        )

    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Không đọc được nội dung ảnh")

    try:
        prediction, _ = get_detector().predict(image)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Dịch vụ mô hình chưa sẵn sàng: {exc}",
        ) from exc

    return {
        "filename": file.filename,
        "rejected": prediction.rejected,
        "reason": prediction.reason,
        "detections": [asdict(item) for item in prediction.detections],
    }

