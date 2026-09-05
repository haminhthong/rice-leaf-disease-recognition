"""Module xử lý và chuẩn hóa nhãn dữ liệu (Annotations Parsing & Normalization).

Module này phụ trách:
1. Chuẩn hóa tên lớp alias từ nhiều bộ dữ liệu nguồn về hai lớp chuẩn.
2. Chuyển đổi định dạng Polygon thành YOLO Bounding Box (x_center, y_center, width, height).
3. Kiểm tra tính hợp lệ của tọa độ ([0, 1], NaN, Inf, lệch biên) và loại bỏ nhãn trùng lặp.
"""

import math
import re
from pathlib import Path

import yaml

from .constants import CLASS_NAMES

Annotation = dict[str, int | float | str]


def normalize_class_name(name: object) -> str | None:
    """Chuẩn hóa tên lớp từ bộ dữ liệu gốc về tên lớp mục tiêu chuẩn.

    Ví dụ:
        - "bacterial leaf blight", "Bacterial LeafBlight" -> "Bacterial_Leaf_Blight"
        - "brown spot", "Brown-Spot" -> "Brown_Spot"

    Args:
        name: Tên lớp nguyên bản từ data.yaml gốc.

    Returns:
        str | None: Tên lớp mục tiêu hoặc None nếu không thuộc hai bệnh cần nhận diện.
    """
    value = re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()
    return {
        "bacterial leaf blight": "Bacterial_Leaf_Blight",
        "bacterial leafblight": "Bacterial_Leaf_Blight",
        "brown spot": "Brown_Spot",
        "brownspot": "Brown_Spot",
    }.get(value)


def build_class_map(dataset_root: Path) -> tuple[list[str], dict[int, int]]:
    """Đọc file `data.yaml` của dataset nguồn và xây dựng bản đồ ánh xạ ID lớp gốc -> ID mục tiêu.

    Args:
        dataset_root: Đường dẫn thư mục gốc của dataset nguồn.

    Returns:
        tuple[list[str], dict[int, int]]: Danh sách tên lớp gốc và dictionary ánh xạ.


    Raises:
        ValueError: Nếu file data.yaml sai định dạng hoặc không chứa đủ lớp mục tiêu.
    """
    yaml_path = dataset_root / "data.yaml"
    with yaml_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Nội dung {yaml_path} không hợp lệ")

    names = config.get("names")
    if isinstance(names, dict):
        try:
            indexed_names = {int(key): value for key, value in names.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Khóa lớp trong {yaml_path} phải là số nguyên") from exc
        expected_ids = list(range(len(indexed_names)))
        if sorted(indexed_names) != expected_ids:
            raise ValueError(f"ID lớp trong {yaml_path} phải liên tục từ 0")
        names = [indexed_names[class_id] for class_id in expected_ids]
    if not isinstance(names, list):
        raise ValueError(f"names trong {dataset_root / 'data.yaml'} không hợp lệ")
    mapping = {}
    for old_id, name in enumerate(names):
        canonical = normalize_class_name(name)
        if canonical in CLASS_NAMES:
            mapping[old_id] = CLASS_NAMES.index(canonical)
    missing = set(range(len(CLASS_NAMES))) - set(mapping.values())
    if missing:
        raise ValueError(f"{dataset_root} thiếu lớp mục tiêu: {missing}; names={names}")
    return names, mapping


def parse_annotation_line(
    line: str,
    class_map: dict[int, int],
    source: Path,
    line_no: int,
) -> tuple[Annotation | None, str | None]:
    """Phân tích một dòng annotation (hỗ trợ cả định dạng YOLO BBox và YOLO Polygon).

    - Nếu là BBox (5 thông số: `class x y w h`), kiểm tra tọa độ chuẩn hóa [0, 1].
    - Nếu là Polygon (>= 7 thông số: `class x1 y1 x2 y2 ...`), tính bounding box bao quanh
      (Bounding Envelope).
    - Cắt bớt phần viền bị xén ngoài khung ảnh ([0, 1]).

    Args:
        line: Chuỗi văn bản đại diện 1 dòng trong file label.
        class_map: Dictionary ánh xạ ID lớp.
        source: Đường dẫn file nhãn nguồn (dùng cho log lỗi).
        line_no: Số thứ tự dòng trong file (1-indexed).

    Returns:
        tuple[Annotation | None, str | None]: Dictionary chứa thông tin annotation hoặc
        thông báo lỗi nếu không hợp lệ.
    """

    parts = line.split()
    if not parts:
        return None, None
    try:
        values = [float(value) for value in parts]
    except ValueError:
        return None, f"{source}:{line_no}: chứa giá trị không phải số"
    raw_class, coordinates = values[0], values[1:]
    if not math.isfinite(raw_class) or not raw_class.is_integer():
        return None, f"{source}:{line_no}: class ID không phải số nguyên"
    old_id = int(raw_class)
    if old_id not in class_map:
        return None, None
    if not coordinates or not all(math.isfinite(v) for v in coordinates):
        return None, f"{source}:{line_no}: tọa độ không hợp lệ"
    tolerance = 1e-6
    if any(v < -tolerance or v > 1 + tolerance for v in coordinates):
        return None, f"{source}:{line_no}: tọa độ ngoài [0, 1]"
    coordinates = [min(1.0, max(0.0, v)) for v in coordinates]
    if len(coordinates) == 4:
        x, y, width, height = coordinates
        source_type = "bbox"
    elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
        xs, ys = coordinates[::2], coordinates[1::2]
        x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
        x, y = (x_min + x_max) / 2, (y_min + y_max) / 2
        width, height, source_type = x_max - x_min, y_max - y_min, "polygon_to_bbox"
    else:
        return None, f"{source}:{line_no}: cần bbox hoặc polygon tối thiểu 3 điểm"
    if width <= 0 or height <= 0:
        return None, f"{source}:{line_no}: width/height phải lớn hơn 0"

    # Đưa hai mép hộp về miền hợp lệ. Một số nhãn nguồn bị lệch do làm tròn.
    original_box = (x, y, width, height)
    x_min = max(0.0, x - width / 2)
    y_min = max(0.0, y - height / 2)
    x_max = min(1.0, x + width / 2)
    y_max = min(1.0, y + height / 2)
    if x_max <= x_min or y_max <= y_min:
        return None, f"{source}:{line_no}: bbox nằm ngoài ảnh"
    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    if not math.isclose(x, original_box[0]) or not math.isclose(y, original_box[1]):
        source_type = f"{source_type}_clipped"
    elif not math.isclose(width, original_box[2]) or not math.isclose(height, original_box[3]):
        source_type = f"{source_type}_clipped"

    return {
        "class_id": class_map[old_id],
        "x": x,
        "y": y,
        "w": width,
        "h": height,
        "source_type": source_type,
    }, None


def parse_label_file(
    path: Path,
    class_map: dict[int, int],
) -> tuple[list[Annotation], list[str], int]:
    """Đọc toàn bộ file nhãn `.txt`, lọc bỏ các dòng lỗi và các annotation bị trùng lặp chính xác.

    Args:
        path: Đường dẫn file `.txt` nhãn YOLO.
        class_map: Dictionary ánh xạ ID lớp.

    Returns:
        tuple[list[Annotation], list[str], int]:
            - Danh sách các annotation hợp lệ duy nhất.
            - Danh sách thông báo lỗi nếu có.
            - Số lượng dòng nhãn bị trùng lặp đã loại bỏ.
    """
    annotations: list[Annotation] = []
    errors: list[str] = []
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            annotation, error = parse_annotation_line(line, class_map, path, line_no)
            if annotation is not None:
                annotations.append(annotation)
            if error is not None:
                errors.append(error)
    unique = {}
    for ann in annotations:
        key = (ann["class_id"], *(round(ann[key], 6) for key in ("x", "y", "w", "h")))
        unique[key] = ann
    return list(unique.values()), errors, len(annotations) - len(unique)
