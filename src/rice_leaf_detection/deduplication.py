"""Thuật toán ngắt trùng lặp và nhóm ảnh gần giống (Exact Deduplication & Near-Duplicate Grouping).

Module này triển khai hai bước quan trọng trong Data Engineering:
1. Loại bỏ ảnh trùng lặp tuyệt đối (Exact Duplicates) dựa trên mã băm SHA-256.
2. Nhóm các ảnh gần giống (Near-Duplicates do Augmentation/Crop) dựa trên khoảng cách Hamming pHash
   sử dụng cấu trúc cây **BK-Tree** (Burkhard-Keller Tree) kết hợp thuật toán Union-Find.

Việc nhóm ảnh này giúp giảm nguy cơ **Data Leakage** bằng cách giữ ảnh trùng và gần trùng
trong cùng một nhóm trước khi chia dữ liệu theo tập Train/Val/Test.
"""

from collections import defaultdict
from typing import Any

Record = dict[str, Any]


class UnionFind:
    """Cấu trúc dữ liệu tập hợp rời rạc Disjoint-Set Union (DSU) với Path Compression và Rank."""

    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        """Tìm đại diện (root) của tập hợp chứa phần tử `value` có áp dụng Path Compression."""
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        """Hợp nhất hai tập hợp chứa `left` và `right` theo Rank."""
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


class BKTree:
    """Cấu trúc cây Burkhard-Keller (BK-Tree) tối ưu hóa truy vấn tìm kiếm phần tử theo metric.

    BK-Tree giúp giảm không gian tìm kiếm (search space) các chuỗi pHash theo khoảng cách Hamming
    so với phương pháp so sánh toàn bộ từng cặp (naive all-pairs comparison) trong nhiều phân bố
    dữ liệu thực tế. Hiệu năng truy vấn thực tế phụ thuộc vào phân bố metric của dữ liệu và không
    đảm bảo cận trên O(log N).
    """

    def __init__(self):
        self.root: list | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        """Tính khoảng cách Hamming giữa 2 chuỗi bit bằng toán tử XOR và đếm bit 1 (bit_count)."""
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        """Thêm một pHash integer kèm chỉ số ảnh vào BK-Tree."""
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            dist = self.distance(value, node[0])
            if dist == 0:
                node[1].append(index)
                return
            if dist not in node[2]:
                node[2][dist] = [value, [index], {}]
                return
            node = node[2][dist]

    def query(self, value: int, maximum: int) -> list[int]:
        """Truy vấn các phần tử trong BK-Tree có khoảng cách Hamming <= `maximum` so với `value`."""
        matches, stack = [], [self.root] if self.root else []
        while stack:
            node = stack.pop()
            dist = self.distance(value, node[0])
            if dist <= maximum:
                matches.extend(node[1])
            stack.extend(
                child for edge, child in node[2].items() if dist - maximum <= edge <= dist + maximum
            )
        return matches


def _signature(record: Record) -> tuple:
    """Tạo chữ ký đại diện cho danh sách annotation của ảnh để phát hiện xung đột nhãn."""
    return tuple(
        sorted(
            (
                ann["class_id"],
                *(round(ann[key], 5) for key in ("x", "y", "w", "h")),
            )
            for ann in record["annotations"]
        )
    )


def deduplicate_and_group(
    records: list[Record],
    phash_distance: int | None,
) -> tuple[list[Record], dict[str, Any]]:
    """Thực hiện lọc trùng SHA-256 và cách ly xung đột nhãn bằng BK-Tree + Union-Find.

    Args:
        records: Danh sách các bản ghi ảnh nguyên bản.
        phash_distance: Ngưỡng khoảng cách Hamming tối đa để kết nối 2 ảnh gần giống nhau.

    Returns:
        tuple[list[Record], dict[str, Any]]:
            - Danh sách các bản ghi ảnh độc lập đã được gán `group_id`.
            - Thống kê chi tiết (ảnh trùng bị loại, xung đột cách ly, liên kết pHash).
    """
    if phash_distance is not None and phash_distance < 0:
        raise ValueError("Khoảng cách pHash không được âm")

    # Loại ảnh trùng tuyệt đối và cách ly nhóm có nhãn xung đột.
    by_sha = defaultdict(list)
    for record in records:
        by_sha[record["sha256"]].append(record)

    unique: list[Record] = []
    quarantined_conflicts: list[dict[str, Any]] = []
    removed = 0

    for digest, group in by_sha.items():
        distinct_signatures = {_signature(item) for item in group}
        if len(distinct_signatures) > 1:
            # Không tự chọn một nhãn khi cùng nội dung ảnh lại có nhiều cách gán nhãn.
            quarantined_conflicts.append(
                {
                    "sha256": digest,
                    "paths": [item["image_path"] for item in group],
                    "reason": "Cùng nội dung ảnh nhưng có các hộp nhãn khác nhau",
                }
            )
            continue

        group.sort(key=lambda item: len(item["annotations"]), reverse=True)
        unique.append(group[0])
        removed += len(group) - 1

    # Giữ các biến thể có cùng tên ảnh gốc trong một nhóm.
    groups = UnionFind(len(unique))
    original_keys = defaultdict(list)
    for index, record in enumerate(unique):
        original_keys[record["original_key"]].append(index)
    for indices in original_keys.values():
        for index in indices[1:]:
            groups.union(indices[0], index)

    # BK-tree giảm số phép so sánh pHash trong các tập dữ liệu phù hợp.
    links = 0
    if phash_distance is not None:
        tree = BKTree()
        for index, record in enumerate(unique):
            value = int(record["phash"], 16)
            for other in tree.query(value, phash_distance):
                if groups.find(index) != groups.find(other):
                    groups.union(index, other)
                    links += 1
            tree.add(value, index)

    # Mỗi thành phần liên thông nhận một mã nhóm ổn định trong lần chạy.
    for index, record in enumerate(unique):
        record["group_id"] = f"group_{groups.find(index):06d}"

    return unique, {
        "exact_duplicates_removed": removed,
        "annotation_conflicts": quarantined_conflicts,
        "near_duplicate_links": links,
    }
