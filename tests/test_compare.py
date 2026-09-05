from pathlib import Path

import pandas as pd

from rice_leaf_detection.compare import REQUIRED_COLUMNS


def test_test_split_not_used_for_selection(tmp_path: Path) -> None:
    """Đảm bảo tập Test hoàn toàn không được dùng để xếp hạng hay đề xuất Champion model."""
    exp_file = tmp_path / "experiments.csv"
    data = [
        {
            "run_name": "model_baseline_val",
            "split": "val",
            "mAP50-95": 0.55,
            "mAP50": 0.80,
            "recall": 0.76,
            "precision": 0.82,
        },
        {
            "run_name": "model_candidate_val",
            "split": "val",
            "mAP50-95": 0.62,
            "mAP50": 0.85,
            "recall": 0.80,
            "precision": 0.86,
        },
        {
            # Entry tập test có chỉ số cao nhưng không được phép tham gia chọn model
            "run_name": "model_candidate_test",
            "split": "test",
            "mAP50-95": 0.99,
            "mAP50": 0.99,
            "recall": 0.99,
            "precision": 0.99,
        },
    ]
    pd.DataFrame(data).to_csv(exp_file, index=False)

    df = pd.read_csv(exp_file)
    assert not (REQUIRED_COLUMNS - set(df.columns))

    val_df = df[df["split"] == "val"].copy()
    val_sorted = val_df.sort_values("mAP50-95", ascending=False)

    champion = val_sorted.iloc[0]["run_name"]
    # Champion phải được chọn từ tập val, không phải từ tập test (dù test có mAP 0.99)
    assert champion == "model_candidate_val"
    assert "test" not in val_sorted["split"].values
