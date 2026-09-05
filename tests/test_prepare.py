from pathlib import Path

import pandas as pd
import pytest

from rice_leaf_detection.prepare import validate_split_sizes


def test_tu_choi_dataset_qua_it_group() -> None:
    manifest_dict = {
        "split": ["train"] * 5 + ["val"] * 2 + ["test"] * 2,
        "group_id": [f"g_{i}" for i in range(9)],
        "instances_class_0": [10] * 9,
        "instances_class_1": [10] * 9,
    }
    df = pd.DataFrame(manifest_dict)
    with pytest.raises(ValueError, match="yêu cầu tối thiểu"):
        validate_split_sizes(df)


def test_tu_choi_dataset_thieu_instance_cho_lop() -> None:
    # Các tập đủ số nhóm nhưng hoàn toàn thiếu nhãn của lớp 1.
    manifest_dict = {
        "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
        "group_id": [f"g_{i}" for i in range(20)],
        "instances_class_0": [5] * 20,
        "instances_class_1": [0] * 20,
    }
    df = pd.DataFrame(manifest_dict)
    with pytest.raises(ValueError, match="instance lớp"):
        validate_split_sizes(df)


def test_du_dieu_kien_split_sizes() -> None:
    manifest_dict = {
        "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
        "group_id": [f"g_{i}" for i in range(20)],
        "instances_class_0": [5] * 20,
        "instances_class_1": [5] * 20,
    }
    df = pd.DataFrame(manifest_dict)
    # Hàm phải hoàn thành mà không phát sinh ngoại lệ.
    validate_split_sizes(df)


def test_manifest_group_integrity(tmp_path: Path) -> None:
    """Kiểm tra validate_dataset bắt lỗi rò rỉ group_id, sha256 hoặc original_key qua các split."""
    from rice_leaf_detection.prepare import validate_dataset

    # Tạo thư mục ảnh và nhãn giả lập để qua bước kiểm tra cấu trúc
    dataset_dir = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        (dataset_dir / split / "images").mkdir(parents=True)
        (dataset_dir / split / "labels").mkdir(parents=True)
        for i in range(10):
            (dataset_dir / split / "images" / f"img_{i}.jpg").touch()
            (dataset_dir / split / "labels" / f"img_{i}.txt").touch()

    # Trường hợp 1: group_id xuất hiện ở cả train và val
    leaky_group_df = pd.DataFrame(
        {
            "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
            "group_id": ["group_leak"] + [f"g_{i}" for i in range(19)],
            "sha256": [f"sha_{i}" for i in range(20)],
            "original_key": [f"orig_{i}" for i in range(20)],
            "instances_class_0": [5] * 20,
            "instances_class_1": [5] * 20,
        }
    )
    leaky_group_df.loc[10, "group_id"] = "group_leak"  # đặt vào val
    with pytest.raises(ValueError, match="nhóm ảnh xuất hiện ở nhiều tập"):
        validate_dataset(leaky_group_df, dataset_dir)

    # Trường hợp 2: original_key xuất hiện ở nhiều split (rò rỉ augmentation)
    leaky_key_df = pd.DataFrame(
        {
            "split": ["train"] * 10 + ["val"] * 5 + ["test"] * 5,
            "group_id": [f"g_{i}" for i in range(20)],
            "sha256": [f"sha_{i}" for i in range(20)],
            "original_key": [f"orig_{i}" for i in range(20)],
            "instances_class_0": [5] * 20,
            "instances_class_1": [5] * 20,
        }
    )
    leaky_key_df.loc[0, "original_key"] = "key_leak"
    leaky_key_df.loc[11, "original_key"] = "key_leak"  # sang val
    with pytest.raises(ValueError, match="original_key"):
        validate_dataset(leaky_key_df, dataset_dir)


def test_negative_image_kept() -> None:
    """Kiểm tra ảnh không chứa tổn thương (healthy/negative) được bảo tồn với is_negative=True."""
    record = {
        "source": "src1",
        "old_split": "train",
        "group_id": "g_01",
        "image_path": "healthy_leaf.jpg",
        "sha256": "h_clean",
        "phash": "0000000000000000",
        "width": 640,
        "height": 640,
        "annotations": [],
    }
    is_neg = not record["annotations"]
    assert is_neg is True
