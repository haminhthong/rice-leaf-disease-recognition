"""RESTful FastAPI Web Service cho Nhận Diện Bệnh Lá Lúa (Rice Leaf Disease Recognition API).

Cung cấp các endpoints:
- `GET /health/live`: Liveness check.
- `GET /health/ready`: Readiness check (kiểm tra khả năng suy luận của mô hình).
- `GET /info`: Thông tin cấu hình và các lớp bệnh được hỗ trợ.
- `POST /predict`: Upload ảnh và nhận kết quả phát hiện bệnh dạng JSON Pydantic Schema.
"""

import asyncio
import logging
from functools import cache
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from rice_leaf_detection import __version__
from rice_leaf_detection.constants import CLASS_NAMES, CLASS_NAMES_VI

from .dependencies import get_detector
from .schemas import DetectionResponse, ImageSummaryResponse, PredictionResponse
from .settings import get_settings
from .validation import decode_and_validate_image

logger = logging.getLogger("rice_leaf_api")

app = FastAPI(
    title="Rice Leaf Disease Detection API",
    description=(
        "API phát hiện tổn thương Bạc lá lúa (Bacterial Leaf Blight) "
        "và Đốm nâu (Brown Spot) bằng YOLOv8. Đây là công cụ hỗ trợ sàng lọc ban đầu."
    ),
    version=__version__,
)

runtime_settings = get_settings()

# Chỉ cho phép các giao diện đã khai báo gọi API từ trình duyệt.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(runtime_settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@cache
def get_semaphore() -> asyncio.Semaphore:
    """Giới hạn số lượt suy luận đồng thời để bảo vệ tài nguyên mô hình."""
    return asyncio.Semaphore(get_settings().inference_concurrency)


@app.get("/health/live", summary="Liveness Probe", tags=["Health"])
def health_live() -> dict[str, str]:
    """Trả về trạng thái hoạt động của tiến trình API server."""
    return {"status": "live"}


@app.get("/health/ready", summary="Readiness Probe", tags=["Health"])
def health_ready() -> dict[str, str]:
    """Kiểm tra mô hình đã được nạp thành công và sẵn sàng suy luận."""
    try:
        detector = get_detector()
        if detector is None:
            raise RuntimeError("Detector không khả dụng")
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        logger.warning("Mô hình chưa sẵn sàng: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ mô hình chưa sẵn sàng suy luận",
        ) from exc
    return {"status": "ready"}


@app.get("/info", summary="Thông tin mô hình", tags=["System"])
def info() -> dict:
    """Trả về phiên bản API, các lớp hỗ trợ và cấu hình suy luận."""
    settings = get_settings()
    return {
        "service": "Rice Leaf Disease Detection API",
        "version": __version__,
        "supported_classes": CLASS_NAMES,
        "supported_classes_vi": CLASS_NAMES_VI,
        "model_weights": settings.weights.as_posix(),
        "image_size": settings.image_size,
        "default_confidence": settings.confidence,
        "default_iou": settings.iou,
        "max_upload_mb": settings.max_upload_bytes // (1024 * 1024),
    }


@app.post("/predict", response_model=PredictionResponse, summary="Dự đoán bệnh", tags=["Inference"])
async def predict(file: Annotated[UploadFile, File()]) -> PredictionResponse:
    """Endpoint tiếp nhận file ảnh upload và trả về danh sách Bounding Boxes phát hiện bệnh.

    - Kiểm tra magic bytes thực tế (JPEG/PNG/WebP).
    - Kiểm tra dung lượng file <= 10MB và tổng số pixels <= 25M.
    - Chạy suy luận trong nhóm luồng để không khóa vòng lặp sự kiện của API.
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

    semaphore = get_semaphore()
    try:
        async with semaphore:
            detector = get_detector()
            prediction, _ = await run_in_threadpool(detector.predict, image)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("Lỗi suy luận mô hình")
        raise HTTPException(
            status_code=503,
            detail="Dịch vụ mô hình suy luận thất bại",
        ) from exc

    image_summary_resp = None
    if prediction.image_summary is not None:
        image_summary_resp = ImageSummaryResponse(
            bacterial_leaf_blight_detected=prediction.image_summary.bacterial_leaf_blight_detected,
            brown_spot_detected=prediction.image_summary.brown_spot_detected,
            total_detections=prediction.image_summary.total_detections,
            requires_human_review=prediction.image_summary.requires_human_review,
            review_reasons=prediction.image_summary.review_reasons,
        )

    return PredictionResponse(
        filename=file.filename,
        status=prediction.status,
        message=prediction.message,
        image_summary=image_summary_resp,
        warnings=prediction.warnings,
        detections=[
            DetectionResponse(
                class_id=d.class_id,
                class_name=d.class_name,
                class_name_vi=d.class_name_vi,
                confidence=d.confidence,
                detection_score=getattr(d, "detection_score", d.confidence),
                box_xyxy=d.box_xyxy,
            )
            for d in prediction.detections
        ],
    )
