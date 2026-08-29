from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    runs_dir: Path


@dataclass(frozen=True)
class DataConfig:
    yaml: Path
    image_size: int


@dataclass(frozen=True)
class ModelConfig:
    weights: str


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_gpu: int
    batch_cpu: int
    patience: int
    workers: int
    optimizer: str
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class InferenceConfig:
    confidence: float
    iou: float


@dataclass(frozen=True)
class ExperimentConfig:
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


def _non_empty(value: str, field: str) -> None:
    if not value.strip():
        raise ValueError(f"Cấu hình '{field}' không được để trống")


def _path(value: object, field: str) -> Path:
    text = str(value).strip()
    _non_empty(text, field)
    return Path(text)


def load_config(path: Path) -> ExperimentConfig:
    """Đọc và kiểm tra toàn bộ cấu hình thí nghiệm từ YAML."""
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
            runs_dir=_path(
                _required(project_raw, "runs_dir", "project"),
                "project.runs_dir",
            ),
        )
        data = DataConfig(
            yaml=_path(_required(data_raw, "yaml", "data"), "data.yaml"),
            image_size=int(_required(data_raw, "image_size", "data")),
        )
        model = ModelConfig(weights=str(_required(model_raw, "weights", "model")))
        training = TrainingConfig(
            epochs=int(_required(training_raw, "epochs", "training")),
            batch_gpu=int(_required(training_raw, "batch_gpu", "training")),
            batch_cpu=int(_required(training_raw, "batch_cpu", "training")),
            patience=int(_required(training_raw, "patience", "training")),
            workers=int(_required(training_raw, "workers", "training")),
            optimizer=str(_required(training_raw, "optimizer", "training")),
            learning_rate=float(
                _required(training_raw, "learning_rate", "training")
            ),
            weight_decay=float(
                _required(training_raw, "weight_decay", "training")
            ),
        )
        inference = InferenceConfig(
            confidence=float(_required(inference_raw, "confidence", "inference")),
            iou=float(_required(inference_raw, "iou", "inference")),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Cấu hình có kiểu dữ liệu không hợp lệ: {exc}") from exc

    _non_negative(project.seed, "project.seed")
    _non_empty(model.weights, "model.weights")
    _non_empty(training.optimizer, "training.optimizer")
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
