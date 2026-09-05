import pytest

from rice_leaf_detection.deduplication import BKTree, deduplicate_and_group


def test_bk_tree_tim_hash_gan_nhau() -> None:
    tree = BKTree()
    tree.add(0b0000, 0)
    tree.add(0b1111, 1)
    assert tree.query(0b0001, 1) == [0]


def test_tu_choi_khoang_cach_am() -> None:
    with pytest.raises(ValueError):
        deduplicate_and_group([], -1)


def test_exact_duplicate_removed() -> None:
    """Kiểm tra loại bỏ chính xác các ảnh trùng SHA-256 có cùng phân bố nhãn."""
    records = [
        {
            "sha256": "hash_abc",
            "phash": "0000000000000000",
            "image_path": "img1.jpg",
            "original_key": "k1",
            "annotations": [{"class_id": 0, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}],
        },
        {
            "sha256": "hash_abc",
            "phash": "0000000000000000",
            "image_path": "img2.jpg",
            "original_key": "k1",
            "annotations": [{"class_id": 0, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}],
        },
    ]
    unique, audit = deduplicate_and_group(records, phash_distance=2)
    assert len(unique) == 1
    assert audit["exact_duplicates_removed"] == 1
    assert len(audit["annotation_conflicts"]) == 0


def test_cach_ly_annotation_conflict_cung_sha256_khac_nhan() -> None:
    records = [
        {
            "sha256": "hash_conflict",
            "phash": "0000000000000000",
            "image_path": "img1.jpg",
            "original_key": "k1",
            "annotations": [{"class_id": 0, "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}],
        },
        {
            "sha256": "hash_conflict",
            "phash": "0000000000000000",
            "image_path": "img2.jpg",
            "original_key": "k1",
            "annotations": [{"class_id": 1, "x": 0.1, "y": 0.1, "w": 0.3, "h": 0.3}],
        },
    ]
    unique, audit = deduplicate_and_group(records, phash_distance=2)
    # Không bản ghi xung đột nào được phép đi vào dữ liệu huấn luyện.
    assert len(unique) == 0
    assert len(audit["annotation_conflicts"]) == 1


def test_near_duplicate_group_not_split() -> None:
    """Kiểm tra các ảnh gần trùng pHash được gom cùng group_id để không bị chia cắt split."""
    records = [
        {
            "sha256": "hash_1",
            "phash": "0000000000000000",
            "image_path": "img1.jpg",
            "original_key": "k1",
            "annotations": [],
        },
        {
            "sha256": "hash_2",
            "phash": "0000000000000001",  # Hamming distance = 1
            "image_path": "img2.jpg",
            "original_key": "k2",
            "annotations": [],
        },
    ]
    unique, audit = deduplicate_and_group(records, phash_distance=2)
    assert len(unique) == 2
    assert audit["near_duplicate_links"] == 1
    # Hai biến thể phải nhận cùng mã nhóm trước khi chia dữ liệu.
    assert unique[0]["group_id"] == unique[1]["group_id"]


def test_same_original_key_not_cross_split() -> None:
    """Kiểm tra các ảnh cùng original_key (biến thể augmentation) được gom cùng group_id."""
    records = [
        {
            "sha256": "hash_x1",
            "phash": "0000000000000000",
            "image_path": "sourceA__leaf_01_crop.jpg",
            "original_key": "sourceA:leaf_01",
            "annotations": [],
        },
        {
            "sha256": "hash_x2",
            "phash": "ffffffffffffffff",  # pHash xa nhau nhưng cùng original_key
            "image_path": "sourceA__leaf_01_rot.jpg",
            "original_key": "sourceA:leaf_01",
            "annotations": [],
        },
    ]
    unique, _ = deduplicate_and_group(records, phash_distance=2)
    assert len(unique) == 2
    assert unique[0]["group_id"] == unique[1]["group_id"]
