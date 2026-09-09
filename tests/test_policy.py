from rice_leaf_detection.constants import OUT_OF_SCOPE_NEGATIVE, TRUE_NEGATIVE
from rice_leaf_detection.policy import choose_policy, score_policy


def test_score_policy_tach_recall_va_bao_dong_gia() -> None:
    rows = [
        {
            "truth_classes": [0],
            "annotation_status": "TARGET_POSITIVE",
            "candidates": [{"class_id": 0, "score": 0.8, "best_iou": 0.7}],
        },
        {
            "truth_classes": [1],
            "annotation_status": "TARGET_POSITIVE",
            "candidates": [{"class_id": 1, "score": 0.35, "best_iou": 0.8}],
        },
        {
            "truth_classes": [],
            "annotation_status": TRUE_NEGATIVE,
            "candidates": [{"class_id": 0, "score": 0.9, "best_iou": 0.0}],
        },
        {
            "truth_classes": [],
            "annotation_status": OUT_OF_SCOPE_NEGATIVE,
            "candidates": [],
        },
    ]

    score = score_policy(rows, review_threshold=0.2, accept_threshold=0.45)

    assert score.metrics["image_target_recall"]["0"]["recall"] == 1.0
    assert score.metrics["image_target_recall"]["1"]["recall"] == 0.0
    assert score.metrics["negative_benchmark"][TRUE_NEGATIVE]["false_alarm_rate"] == 1.0


def test_choose_policy_chi_dung_candidate_validation() -> None:
    rows = [
        {
            "truth_classes": [0],
            "annotation_status": "TARGET_POSITIVE",
            "candidates": [{"class_id": 0, "score": 0.8, "best_iou": 0.8}],
        },
        {
            "truth_classes": [],
            "annotation_status": TRUE_NEGATIVE,
            "candidates": [],
        },
    ]

    selected, ranking = choose_policy(
        rows,
        preferred_review_threshold=0.2,
        preferred_accept_threshold=0.45,
        review_values=(0.2, 0.4),
        accept_values=(0.45, 0.8),
    )

    assert selected.accept_threshold in {0.45, 0.8}
    assert ranking
    assert all(candidate.review_threshold <= candidate.accept_threshold for candidate in ranking)
