import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from rice_leaf_detection.inference import RiceLeafDetector

app = FastAPI(
    title="Rice Leaf Disease Detection API",
    description="Phát hiện bạc lá và đốm nâu trên ảnh lá lúa.",
    version="1.0.0",
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@lru_cache(maxsize=1)
def get_detector() -> RiceLeafDetector:
    weights = Path(os.getenv("RICE_MODEL_PATH", "artifacts/best.pt"))
    confidence = float(os.getenv("RICE_CONFIDENCE", "0.25"))
    iou = float(os.getenv("RICE_IOU", "0.45"))
    return RiceLeafDetector(weights, confidence=confidence, iou=iou)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Chỉ hỗ trợ JPEG, PNG hoặc WebP")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh vượt quá giới hạn 10 MB")
    import cv2
    import numpy as np

    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Không đọc được nội dung ảnh")
    try:
        prediction, _ = get_detector().predict(image)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "filename": file.filename,
        "rejected": prediction.rejected,
        "reason": prediction.reason,
        "detections": [asdict(item) for item in prediction.detections],
    }
