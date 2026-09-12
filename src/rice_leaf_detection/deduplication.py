"""Lọc trùng tuyệt đối (SHA-256) và gom nhóm ảnh biến thể (pHash Hamming distance).

Module này giải quyết vấn đề Data Leakage bằng hai bước:
1. Loại bỏ các ảnh trùng lặp tuyệt đối (Exact Duplicates) dựa trên mã băm SHA-256.
   Nếu cùng nội dung ảnh nhưng có nhãn gán khác nhau (conflict), ảnh sẽ được cách ly.
2. Gom nhóm các ảnh biến thể (Near-Duplicates do Augmentation/Crop hoặc cùng ảnh gốc)
   vào cùng một group_id để toàn bộ nhóm được giữ trọn vẹn trong một split (Train, Val hoặc Test).
"""

from collections import defaultdict
from typing import Any

Record = dict[str, Any]


def hamming_distance(h1: str | int, h2: str | int) -> int:
    """Tính khoảng cách Hamming giữa 2 mã băm pHash 64-bit (số bit khác nhau)."""
    val1 = int(h1, 16) if isinstance(h1, str) else h1
    val2 = int(h2, 16) if isinstance(h2, str) else h2
    return (val1 ^ val2).bit_count()


class DisjointSet:
    """Tập hợp rời rạc đơn giản (Union-Find) để gom nhóm các ảnh liên quan."""

    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        root_i, root_j = self.find(i), self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i


def _signature(record: Record) -> tuple:
    """Tạo chữ ký đại diện cho danh sách annotation để phát hiện xung đột nhãn."""
    return tuple(
        sorted(
            (
                ann["class_id"],
                *(round(ann[key], 5) for key in ("x", "y", "w", "h")),
            )
            for ann in record.get("annotations", [])
        )
    )


def deduplicate_and_group(
    records: list[Record],
    phash_distance: int | None = 2,
) -> tuple[list[Record], dict[str, Any]]:
    """Lọc trùng SHA-256 và gom nhóm ảnh biến thể chống rò rỉ dữ liệu.

    Args:
        records: Danh sách các bản ghi ảnh thô.
        phash_distance: Ngưỡng khoảng cách Hamming tối đa để liên kết 2 ảnh gần giống nhau.

    Returns:
        tuple[list[Record], dict[str, Any]]:
            - Danh sách các bản ghi ảnh sạch đã gán group_id.
            - Thống kê (số ảnh trùng bị loại, xung đột nhãn, liên kết pHash).
    """
    if phash_distance is not None and phash_distance < 0:
        raise ValueError("Khoảng cách pHash không được âm")

    # 1. Lọc trùng lặp tuyệt đối theo SHA-256
    by_sha = defaultdict(list)
    for record in records:
        by_sha[record["sha256"]].append(record)

    unique: list[Record] = []
    annotation_conflicts: list[dict[str, Any]] = []
    removed_count = 0

    for digest, group in by_sha.items():
        distinct_signatures = {_signature(item) for item in group}
        if len(distinct_signatures) > 1:
            # Cùng nội dung ảnh nhưng nhãn gán mâu thuẫn -> cách ly kiểm duyệt
            annotation_conflicts.append(
                {
                    "sha256": digest,
                    "paths": [item["image_path"] for item in group],
                    "reason": "Cùng nội dung ảnh nhưng có các hộp nhãn khác nhau",
                }
            )
            continue

        # Giữ lại 1 bản ghi đại diện có nhiều annotation nhất
        group.sort(key=lambda item: len(item.get("annotations", [])), reverse=True)
        unique.append(group[0])
        removed_count += len(group) - 1

    n = len(unique)
    groups = DisjointSet(n)

    # 2. Gom nhóm các ảnh có cùng original_key (ảnh gốc trước khi crop/xoay)
    by_original_key = defaultdict(list)
    for idx, record in enumerate(unique):
        orig_key = record.get("original_key")
        if orig_key:
            by_original_key[orig_key].append(idx)

    for indices in by_original_key.values():
        for idx in indices[1:]:
            groups.union(indices[0], idx)

    # 3. Kiểm tra near-duplicate bằng pHash Hamming distance
    near_duplicate_links = 0
    if phash_distance is not None and phash_distance >= 0 and n > 1:
        phash_ints = [int(rec["phash"], 16) for rec in unique]
        for i in range(n):
            for j in range(i + 1, n):
                if groups.find(i) != groups.find(j):
                    dist = (phash_ints[i] ^ phash_ints[j]).bit_count()
                    if dist <= phash_distance:
                        groups.union(i, j)
                        near_duplicate_links += 1

    # Gán group_id duy nhất cho từng nhóm liên thông
    for idx, record in enumerate(unique):
        record["group_id"] = f"group_{groups.find(idx):06d}"

    return unique, {
        "exact_duplicates_removed": removed_count,
        "annotation_conflicts": annotation_conflicts,
        "near_duplicate_links": near_duplicate_links,
    }
