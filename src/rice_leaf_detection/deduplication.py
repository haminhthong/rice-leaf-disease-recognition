from collections import defaultdict
from typing import Any

Record = dict[str, Any]


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


class BKTree:
    def __init__(self):
        self.root: list | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = [value, [index], {}]
            return
        node = self.root
        while True:
            distance = self.distance(value, node[0])
            if distance == 0:
                node[1].append(index)
                return
            if distance not in node[2]:
                node[2][distance] = [value, [index], {}]
                return
            node = node[2][distance]

    def query(self, value: int, maximum: int) -> list[int]:
        matches, stack = [], [self.root] if self.root else []
        while stack:
            node = stack.pop()
            distance = self.distance(value, node[0])
            if distance <= maximum:
                matches.extend(node[1])
            stack.extend(child for edge, child in node[2].items()
                         if distance - maximum <= edge <= distance + maximum)
        return matches


def _signature(record: Record) -> tuple:
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
    if phash_distance is not None and phash_distance < 0:
        raise ValueError("Khoảng cách pHash không được âm")

    by_sha = defaultdict(list)
    for record in records:
        by_sha[record["sha256"]].append(record)
    unique, conflicts, removed = [], [], 0
    for digest, group in by_sha.items():
        group.sort(key=lambda item: len(item["annotations"]), reverse=True)
        unique.append(group[0])
        removed += len(group) - 1
        if len({_signature(item) for item in group}) > 1:
            conflicts.append(
                {
                    "sha256": digest,
                    "paths": [item["image_path"] for item in group],
                    "chosen": group[0]["image_path"],
                }
            )

    groups = UnionFind(len(unique))
    original_keys = defaultdict(list)
    for index, record in enumerate(unique):
        original_keys[record["original_key"]].append(index)
    for indices in original_keys.values():
        for index in indices[1:]:
            groups.union(indices[0], index)

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
    for index, record in enumerate(unique):
        record["group_id"] = f"group_{groups.find(index):06d}"
    return unique, {
        "exact_duplicates_removed": removed,
        "annotation_conflicts": conflicts,
        "near_duplicate_links": links,
    }
