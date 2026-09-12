"""Quản lý và kiểm tra cấu hình thí nghiệm từ YAML bằng Dataclass bất biến."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    """Cấu hình chung dự án (seed tái lập, thư mục lưu runs)."""

    seed: int
    runs_dir: Path


@dataclass(frozen=True)
class DataConfig:
    """Cấu hình bộ dữ liệu (đường dẫn file data.yaml, kích thước ảnh)."""

    yaml: Path
    image_size: int


@dataclass(frozen=True)
class ModelConfig:
    """Cấu hình mô hình YOLOv8."""

    architecture: str
    weights: str


@dataclass(frozen=True)
class AugmentationConfig:
    """Chính sách augmentation áp dụng cho tập Train."""

    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    translate: float
    scale: float
    fliplr: float
    flipud: float
    mosaic: float
    mixup: float
    close_mosaic: int


@dataclass(frozen=True)
class TrainingConfig:
    """Cấu hình tham số huấn luyện mô hình."""

    epochs: int
    batch_gpu: int
    batch_cpu: int
    patience: int
    workers: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    augmentation: AugmentationConfig


@dataclass(frozen=True)
class InferenceConfig:
    """Cấu hình ngưỡng tin cậy và NMS IoU."""

    confidence: float
    iou: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Tổng hợp toàn bộ cấu hình cho một thí nghiệm."""

    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Cấu hình '{field}' phải là một ánh xạ YAML")
    return value


def _required(mapping: dict[str, Any], key: str, group: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Thiếu khóa cấu hình: {group}.{key}")
    return mapping[key]


def _positive(value: int | float, field: str) -> None:
    if value <= 0:
        raise ValueError(f"Cấu hình '{field}' phải lớn hơn 0")


def _non_negative(value: int | float, field: str) -> None:
    if value < 0:
        raise ValueError(f"Cấu hình '{field}' không được âm")


def _probability(value: float, field: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"Cấu hình '{field}' phải nằm trong khoảng [0, 1]")


def _text(value: object, field: str) -> str:
    if value is None:
        raise ValueError(f"Cấu hình '{field}' không được để trống")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Cấu hình '{field}' không được để trống")
    return text


def load_config(path: Path) -> ExperimentConfig:
    """Đọc, xác thực và ánh xạ cấu hình từ file YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    root = _mapping(raw, "gốc")

    project_raw = _mapping(_required(root, "project", "gốc"), "project")
    data_raw = _mapping(_required(root, "data", "gốc"), "data")
    model_raw = _mapping(_required(root, "model", "gốc"), "model")
    training_raw = _mapping(_required(root, "training", "gốc"), "training")
    inference_raw = _mapping(_required(root, "inference", "gốc"), "inference")

    try:
        project = ProjectConfig(
            seed=int(_required(project_raw, "seed", "project")),
            runs_dir=Path(_text(_required(project_raw, "runs_dir", "project"), "project.runs_dir")),
        )
        data = DataConfig(
            yaml=Path(_text(_required(data_raw, "yaml", "data"), "data.yaml")),
            image_size=int(_required(data_raw, "image_size", "data")),
        )
        model = ModelConfig(
            architecture=_text(model_raw.get("architecture", "yolov8s"), "model.architecture"),
            weights=_text(_required(model_raw, "weights", "model"), "model.weights"),
        )
        augmentation = AugmentationConfig(
            hsv_h=float(training_raw.get("hsv_h", 0.005)),
            hsv_s=float(training_raw.get("hsv_s", 0.25)),
            hsv_v=float(training_raw.get("hsv_v", 0.20)),
            degrees=float(training_raw.get("degrees", 10.0)),
            translate=float(training_raw.get("translate", 0.05)),
            scale=float(training_raw.get("scale", 0.15)),
            fliplr=float(training_raw.get("fliplr", 0.50)),
            flipud=float(training_raw.get("flipud", 0.00)),
            mosaic=float(training_raw.get("mosaic", 0.20)),
            mixup=float(training_raw.get("mixup", 0.00)),
            close_mosaic=int(training_raw.get("close_mosaic", 10)),
        )
        training = TrainingConfig(
            epochs=int(_required(training_raw, "epochs", "training")),
            batch_gpu=int(_required(training_raw, "batch_gpu", "training")),
            batch_cpu=int(_required(training_raw, "batch_cpu", "training")),
            patience=int(_required(training_raw, "patience", "training")),
            workers=int(_required(training_raw, "workers", "training")),
            optimizer=_text(_required(training_raw, "optimizer", "training"), "training.optimizer"),
            learning_rate=float(_required(training_raw, "learning_rate", "training")),
            weight_decay=float(_required(training_raw, "weight_decay", "training")),
            augmentation=augmentation,
        )
        inference = InferenceConfig(
            confidence=float(
                inference_raw.get("confidence", inference_raw.get("candidate_confidence", 0.45))
            ),
            iou=float(_required(inference_raw, "iou", "inference")),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cấu hình có kiểu dữ liệu không hợp lệ: {exc}") from exc

    _non_negative(project.seed, "project.seed")
    _positive(data.image_size, "data.image_size")
    _positive(training.epochs, "training.epochs")
    _positive(training.batch_gpu, "training.batch_gpu")
    _positive(training.batch_cpu, "training.batch_cpu")
    _non_negative(training.patience, "training.patience")
    _non_negative(training.workers, "training.workers")
    _positive(training.learning_rate, "training.learning_rate")
    _non_negative(training.weight_decay, "training.weight_decay")
    _probability(inference.confidence, "inference.confidence")
    _probability(inference.iou, "inference.iou")

    return ExperimentConfig(project, data, model, training, inference)
