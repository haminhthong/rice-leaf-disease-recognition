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

