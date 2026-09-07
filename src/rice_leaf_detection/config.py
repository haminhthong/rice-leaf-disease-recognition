"""Quản lý và kiểm tra cấu hình thí nghiệm từ YAML bằng Dataclass bất biến.

Module này ánh xạ file cấu hình YAML thành các dataclass bất biến (frozen dataclasses),
thực hiện kiểm tra nghiêm ngặt kiểu dữ liệu, các thuộc tính bắt buộc và giới hạn miền giá trị
trước khi khởi chạy các công đoạn huấn luyện hoặc suy luận.
"""

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
    """Cấu hình trọng số mô hình YOLOv8."""

    architecture: str
    weights: str


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
    augmentation: "AugmentationConfig"


@dataclass(frozen=True)
class AugmentationConfig:
    """Chính sách augmentation chỉ áp dụng cho tập Train."""

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
class InferenceConfig:
    """Cấu hình lấy candidate từ detector và ngưỡng NMS."""

    candidate_confidence: float
    iou: float

    @property
    def confidence(self) -> float:
        """Tên cũ để các integration bên ngoài không bị hỏng đột ngột."""
        return self.candidate_confidence


@dataclass(frozen=True)
class DecisionPolicyConfig:
    """Ngưỡng quyết định nghiệp vụ được chọn trên Validation."""

    review_threshold: float
    accept_threshold: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Tổng hợp toàn bộ cấu hình cho một thí nghiệm."""

    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    policy: DecisionPolicyConfig


def _mapping(value: object, field: str) -> dict[str, Any]:
    """Kiểm tra giá trị đầu vào có đúng là dictionary/mapping hay không."""
    if not isinstance(value, dict):
        raise ValueError(f"Cấu hình '{field}' phải là một ánh xạ YAML")
    return value


def _required(mapping: dict[str, Any], key: str, group: str) -> Any:
    """Đảm bảo khóa bắt buộc tồn tại trong nhóm cấu hình."""
    if key not in mapping:
        raise ValueError(f"Thiếu khóa cấu hình: {group}.{key}")
    return mapping[key]


def _positive(value: int | float, field: str) -> None:
    """Kiểm tra giá trị phải là số dương (> 0)."""
    if value <= 0:
        raise ValueError(f"Cấu hình '{field}' phải lớn hơn 0")


def _non_negative(value: int | float, field: str) -> None:
    """Kiểm tra giá trị không được là số âm (>= 0)."""
    if value < 0:
        raise ValueError(f"Cấu hình '{field}' không được âm")


def _probability(value: float, field: str) -> None:
    """Kiểm tra giá trị phải nằm trong khoảng xác suất [0, 1]."""
    if not 0 <= value <= 1:
        raise ValueError(f"Cấu hình '{field}' phải nằm trong khoảng [0, 1]")


def _non_empty(value: str, field: str) -> None:
    """Kiểm tra chuỗi văn bản không được rỗng hoặc chứa toàn khoảng trắng."""
    if not value.strip():
        raise ValueError(f"Cấu hình '{field}' không được để trống")


def _text(value: object, field: str) -> str:
    """Chuẩn hóa trường văn bản bắt buộc và từ chối giá trị None/rỗng."""
    if value is None:
        raise ValueError(f"Cấu hình '{field}' không được để trống")
    text = str(value).strip()
    _non_empty(text, field)
    return text


def _path(value: object, field: str) -> Path:
    """Chuyển đổi chuỗi thành đối tượng Path."""
    return Path(_text(value, field))


def load_config(path: Path) -> ExperimentConfig:
    """Đọc, validate và ánh xạ toàn bộ cấu hình thí nghiệm từ file YAML.

    Args:
        path: Đường dẫn tới file cấu hình YAML.

    Returns:
        ExperimentConfig: Đối tượng cấu hình bất biến đã được kiểm tra tính hợp lệ.

    Raises:
        FileNotFoundError: Nếu không tìm thấy file cấu hình tại đường dẫn chỉ định.
        ValueError: Nếu cấu hình thiếu trường, sai kiểu dữ liệu hoặc ngoài phạm vi cho phép.
    """
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
    policy_raw = _mapping(root.get("policy", {}), "policy")

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
        model = ModelConfig(
            architecture=_text(
                model_raw.get("architecture", "yolov8s"), "model.architecture"
            ),
            weights=_text(_required(model_raw, "weights", "model"), "model.weights")
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
            optimizer=_text(
                _required(training_raw, "optimizer", "training"),
                "training.optimizer",
            ),
            learning_rate=float(_required(training_raw, "learning_rate", "training")),
            weight_decay=float(_required(training_raw, "weight_decay", "training")),
            augmentation=augmentation,
        )
        inference = InferenceConfig(
            candidate_confidence=float(
                inference_raw.get("candidate_confidence", inference_raw.get("confidence", 0.20))
            ),
            iou=float(_required(inference_raw, "iou", "inference")),
        )
        policy = DecisionPolicyConfig(
            review_threshold=float(policy_raw.get("review_threshold", 0.20)),
            accept_threshold=float(policy_raw.get("accept_threshold", 0.45)),
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
    for field_name in ("hsv_h", "hsv_s", "hsv_v", "translate", "scale"):
        _non_negative(getattr(training.augmentation, field_name), f"training.{field_name}")
    _non_negative(training.augmentation.degrees, "training.degrees")
    for field_name in ("fliplr", "flipud", "mosaic", "mixup"):
        _probability(getattr(training.augmentation, field_name), f"training.{field_name}")
    _non_negative(training.augmentation.close_mosaic, "training.close_mosaic")
    _probability(inference.candidate_confidence, "inference.candidate_confidence")
    _probability(inference.iou, "inference.iou")
    _probability(policy.review_threshold, "policy.review_threshold")
    _probability(policy.accept_threshold, "policy.accept_threshold")
    if policy.review_threshold > policy.accept_threshold:
        raise ValueError("policy.review_threshold không được lớn hơn policy.accept_threshold")
    return ExperimentConfig(project, data, model, training, inference, policy)
