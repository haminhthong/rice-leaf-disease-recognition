"""Hiệu chỉnh decision policy trên Validation và tạo artifact có thể truy vết.

Module này tách phần tính điểm policy khỏi YOLO để có thể kiểm thử không cần
GPU hay trọng số thật. Validation chỉ dùng để chọn hai ngưỡng:

* ``review_threshold``: ngưỡng giữ candidate để chuyển người kiểm tra.
* ``accept_threshold``: ngưỡng chấp nhận candidate ở mức ảnh.

Tập Test không được đưa vào bất kỳ hàm chọn ngưỡng nào.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import OUT_OF_SCOPE_NEGATIVE, TRUE_NEGATIVE
from .error_analysis import box_iou, read_yolo_labels


@dataclass(frozen=True)
class PolicyScore:
    """Điểm đánh giá của một cặp ngưỡng trên một tập dữ liệu."""

    review_threshold: float
    accept_threshold: float
    macro_image_recall: float
    macro_false_alarm_rate: float
    macro_review_recall: float
    macro_review_false_alarm_rate: float
    objective: float
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Chuyển kết quả thành dictionary JSON-serializable."""
        return asdict(self)


def _accepted_candidates(
    row: dict[str, Any],
    accept_threshold: float,
) -> list[dict[str, Any]]:
    """Lấy các candidate đạt ngưỡng chấp nhận trong một ảnh."""
    return [
        candidate
        for candidate in row.get("candidates", [])
        if float(candidate["score"]) >= accept_threshold
    ]


def score_policy(
    rows: Iterable[dict[str, Any]],
    review_threshold: float,
    accept_threshold: float,
    iou_threshold: float = 0.5,
) -> PolicyScore:
    """Tính recall theo ảnh và false-alarm rate cho một cặp ngưỡng.

    ``rows`` phải chứa ``truth_classes``, ``annotation_status`` và danh sách
    candidate. Mỗi candidate cần có ``class_id``, ``score``, ``box_xyxy`` và
    ``best_iou``. ``best_iou`` là IoU tốt nhất với ground truth cùng lớp.
    """
    if not 0 <= review_threshold <= 1:
        raise ValueError("review_threshold phải nằm trong [0, 1]")
    if not 0 <= accept_threshold <= 1:
        raise ValueError("accept_threshold phải nằm trong [0, 1]")
    if review_threshold > accept_threshold:
        raise ValueError("review_threshold không được lớn hơn accept_threshold")
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold phải nằm trong (0, 1]")

    positive_total = {0: 0, 1: 0}
    positive_recalled = {0: 0, 1: 0}
    review_recalled = {0: 0, 1: 0}
    negative_total = {TRUE_NEGATIVE: 0, OUT_OF_SCOPE_NEGATIVE: 0}
    negative_false_alarm = {TRUE_NEGATIVE: 0, OUT_OF_SCOPE_NEGATIVE: 0}
    review_negative_total = {TRUE_NEGATIVE: 0, OUT_OF_SCOPE_NEGATIVE: 0}
    review_negative_false_alarm = {TRUE_NEGATIVE: 0, OUT_OF_SCOPE_NEGATIVE: 0}

    for row in rows:
        truth_classes = {int(value) for value in row.get("truth_classes", [])}
        accepted = _accepted_candidates(row, accept_threshold)
        review_candidates = [
            candidate
            for candidate in row.get("candidates", [])
            if float(candidate["score"]) >= review_threshold
        ]
        recalled_classes = {
            int(candidate["class_id"])
            for candidate in accepted
            if float(candidate.get("best_iou", 0.0)) >= iou_threshold
        }
        review_recalled_classes = {
            int(candidate["class_id"])
            for candidate in review_candidates
            if float(candidate.get("best_iou", 0.0)) >= iou_threshold
        }
        for class_id in truth_classes & positive_total.keys():
            positive_total[class_id] += 1
            if class_id in recalled_classes:
                positive_recalled[class_id] += 1
            if class_id in review_recalled_classes:
                review_recalled[class_id] += 1

        status = row.get("annotation_status")
        if status in negative_total:
            negative_total[status] += 1
            if accepted:
                negative_false_alarm[status] += 1
            review_negative_total[status] += 1
            if review_candidates:
                review_negative_false_alarm[status] += 1

    recall_values = [
        positive_recalled[class_id] / positive_total[class_id]
        for class_id in (0, 1)
        if positive_total[class_id]
    ]
    false_alarm_values = [
        negative_false_alarm[status] / negative_total[status]
        for status in (TRUE_NEGATIVE, OUT_OF_SCOPE_NEGATIVE)
        if negative_total[status]
    ]
    macro_recall = sum(recall_values) / len(recall_values) if recall_values else 0.0
    macro_false_alarm = (
        sum(false_alarm_values) / len(false_alarm_values) if false_alarm_values else 0.0
    )
    review_recall_values = [
        review_recalled[class_id] / positive_total[class_id]
        for class_id in (0, 1)
        if positive_total[class_id]
    ]
    review_false_alarm_values = [
        review_negative_false_alarm[status] / review_negative_total[status]
        for status in (TRUE_NEGATIVE, OUT_OF_SCOPE_NEGATIVE)
        if review_negative_total[status]
    ]
    macro_review_recall = (
        sum(review_recall_values) / len(review_recall_values) if review_recall_values else 0.0
    )
    macro_review_false_alarm = (
        sum(review_false_alarm_values) / len(review_false_alarm_values)
        if review_false_alarm_values
        else 0.0
    )
    metrics = {
        "image_target_recall": {
            str(class_id): {
                "positive_images": positive_total[class_id],
                "recalled_images": positive_recalled[class_id],
                "recall": (
                    round(positive_recalled[class_id] / positive_total[class_id], 4)
                    if positive_total[class_id]
                    else None
                ),
            }
            for class_id in (0, 1)
        },
        "negative_benchmark": {
            status: {
                "negative_images": negative_total[status],
                "images_with_accepted_detection": negative_false_alarm[status],
                "false_alarm_rate": (
                    round(negative_false_alarm[status] / negative_total[status], 4)
                    if negative_total[status]
                    else None
                ),
            }
            for status in (TRUE_NEGATIVE, OUT_OF_SCOPE_NEGATIVE)
        },
        "review_benchmark": {
            "macro_recall": round(macro_review_recall, 4),
            "macro_false_alarm_rate": round(macro_review_false_alarm, 4),
        },
    }
    objective = (
        macro_recall
        - macro_false_alarm
        + 0.10 * macro_review_recall
        - 0.25 * macro_review_false_alarm
    )
    return PolicyScore(
        review_threshold=review_threshold,
        accept_threshold=accept_threshold,
        macro_image_recall=round(macro_recall, 6),
        macro_false_alarm_rate=round(macro_false_alarm, 6),
        macro_review_recall=round(macro_review_recall, 6),
        macro_review_false_alarm_rate=round(macro_review_false_alarm, 6),
        objective=round(objective, 6),
        metrics=metrics,
    )


def choose_policy(
    rows: Iterable[dict[str, Any]],
    preferred_review_threshold: float,
    preferred_accept_threshold: float,
    review_values: Iterable[float] | None = None,
    accept_values: Iterable[float] | None = None,
    iou_threshold: float = 0.5,
) -> tuple[PolicyScore, list[PolicyScore]]:
    """Tìm policy tốt nhất trên Validation theo objective recall - false alarm.

    Khi nhiều candidate có cùng objective, ưu tiên recall cao hơn, false alarm
    thấp hơn và cuối cùng policy gần với giá trị cấu hình ban đầu. Nhờ vậy kết
    quả ổn định hơn trên tập validation nhỏ.
    """
    row_list = list(rows)
    review_grid = list(review_values or (round(value / 100, 2) for value in range(10, 51, 5)))
    accept_grid = list(accept_values or (round(value / 100, 2) for value in range(30, 96, 5)))
    scores = [
        score
        for review in review_grid
        for accept in accept_grid
        if review <= accept
        for score in [score_policy(row_list, review, accept, iou_threshold)]
    ]
    if not scores:
        raise ValueError("Không tạo được candidate policy hợp lệ")

    def ranking(score: PolicyScore) -> tuple[float, float, float, float, float]:
        distance = abs(score.review_threshold - preferred_review_threshold) + abs(
            score.accept_threshold - preferred_accept_threshold
        )
        return (
            score.objective,
            score.macro_image_recall,
            -score.macro_false_alarm_rate,
            -distance,
            score.accept_threshold,
        )

    selected = max(scores, key=ranking)
    return selected, sorted(scores, key=ranking, reverse=True)


def collect_policy_rows(
    model: Any,
    dataset_dir: Path,
    manifest: Any,
    split: str = "val",
    confidence: float = 0.001,
    iou: float = 0.7,
    image_size: int = 640,
) -> list[dict[str, Any]]:
    """Chạy model một lần trên Validation để tạo dữ liệu cho grid search policy."""
    if not 0 <= confidence <= 1:
        raise ValueError("confidence phải nằm trong [0, 1]")
    rows: list[dict[str, Any]] = []
    for record in manifest[manifest["split"] == split].itertuples(index=False):
        image_path = dataset_dir / record.output_image
        label_path = dataset_dir / split / "labels" / f"{image_path.stem}.txt"
        truth = read_yolo_labels(label_path, int(record.width), int(record.height))
        result = model.predict(
            source=str(image_path),
            imgsz=image_size,
            conf=confidence,
            iou=iou,
            verbose=False,
        )[0]
        candidates: list[dict[str, Any]] = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            box_xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
            same_class_ious = [
                box_iou(box_xyxy, true_box)
                for true_class, true_box, _ in truth
                if true_class == class_id
            ]
            candidates.append(
                {
                    "class_id": class_id,
                    "score": float(box.conf[0]),
                    "box_xyxy": box_xyxy,
                    "best_iou": max(same_class_ious, default=0.0),
                }
            )
        rows.append(
            {
                "truth_classes": [class_id for class_id, _, _ in truth],
                "annotation_status": getattr(record, "annotation_status", None),
                "candidates": candidates,
            }
        )
    if not rows:
        raise ValueError(f"Không có record trong split={split} để tune policy")
    return rows


def write_policy_report(
    path: Path,
    selected: PolicyScore,
    ranking: list[PolicyScore],
    model_path: Path,
    dataset_manifest_hash: str | None,
) -> None:
    """Ghi báo cáo hiệu chỉnh policy và top candidate ra JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": "val",
        "objective": (
            "macro_image_recall - macro_false_alarm_rate + "
            "0.10*macro_review_recall - 0.25*macro_review_false_alarm_rate"
        ),
        "selected_policy": {
            "review_threshold": selected.review_threshold,
            "accept_threshold": selected.accept_threshold,
        },
        "selected_score": selected.to_dict(),
        "top_candidates": [score.to_dict() for score in ranking[:10]],
        "model": str(model_path.resolve()),
        "dataset_manifest_sha256": dataset_manifest_hash,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
