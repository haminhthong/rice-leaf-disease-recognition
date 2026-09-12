"""RESTful FastAPI Web Service cho Nhận Diện Bệnh Lá Lúa.

Cung cấp 2 endpoints chuẩn:
- `GET /health`: Healthcheck xác nhận dịch vụ hoạt động.
- `POST /predict`: Upload ảnh và nhận kết quả phát hiện tổn thương dạng JSON Pydantic.
"""

import logging
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from rice_leaf_detection import __version__

from .dependencies import get_detector
from .schemas import DetectionResponse, PredictionResponse
from .settings import get_settings
from .validation import decode_and_validate_image

logger = logging.getLogger("rice_leaf_api")

app = FastAPI(
    title="Rice Leaf Disease Detection API",
    description="API phát hiện tổn thương Bạc lá lúa và Đốm nâu bằng YOLOv8.",
    version=__version__,
)

runtime_settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(runtime_settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health Check", tags=["System"])
def health() -> dict[str, str]:
    """Kiểm tra trạng thái sẵn sàng hoạt động của API server."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Phát hiện bệnh lá lúa",
    tags=["Inference"],
)
async def predict(file: Annotated[UploadFile, File()]) -> PredictionResponse:
    """Tiếp nhận ảnh lá lúa tải lên và trả về danh sách các vùng tổn thương phát hiện được.

    - Kiểm tra magic bytes ảnh (JPEG, PNG, WebP).
    - Giới hạn dung lượng tối đa 10 MB.
    """
    settings = get_settings()
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        limit_mb = settings.max_upload_bytes // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Dung lượng ảnh vượt quá giới hạn {limit_mb} MB.",
        )

    image = decode_and_validate_image(content, max_pixels=settings.max_image_pixels)

    try:
        detector = get_detector()
        prediction, _ = detector.predict(image)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("Lỗi khi suy luận mô hình")
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ mô hình suy luận thất bại hoặc chưa có trọng số.",
        ) from exc

    return PredictionResponse(
        filename=file.filename,
        status=prediction.status,
        message=prediction.message,
        warnings=prediction.warnings,
        detections=[
            DetectionResponse(
                class_id=d.class_id,
                class_name=d.class_name,
                class_name_vi=d.class_name_vi,
                confidence=d.confidence,
                box_xyxy=d.box_xyxy,
            )
            for d in prediction.detections
        ],
    )
