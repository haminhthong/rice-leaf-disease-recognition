from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    """Đọc cấu hình YAML và kiểm tra các nhóm bắt buộc."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình trong {path} phải là một ánh xạ YAML")
    required = {"project", "data", "model", "training", "inference"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Cấu hình thiếu các nhóm: {sorted(missing)}")
    return deepcopy(config)


def nested_value(config: dict[str, Any], *keys: str) -> Any:
    """Lấy giá trị lồng nhau và báo lỗi rõ ràng khi thiếu khóa."""
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError("Thiếu khóa cấu hình: " + ".".join(keys))
        value = value[key]
    return value

