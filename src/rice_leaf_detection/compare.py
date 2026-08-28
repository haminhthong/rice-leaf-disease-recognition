import argparse
from pathlib import Path

import pandas as pd

from .utils import configure_utf8_console


REQUIRED_COLUMNS = {"run_name", "split", "mAP50-95", "mAP50", "recall", "precision"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Xếp hạng mô hình bằng tập xác thực")
    parser.add_argument("--experiments", type=Path, default=Path("runs/evaluate/experiments.csv"))
    parser.add_argument(
        "--metric",
        choices=("mAP50-95", "mAP50", "recall", "precision"),
        default="mAP50-95",
    )
    return parser.parse_args()


def main() -> None:
    configure_utf8_console()
    args = parse_args()
    if not args.experiments.exists():
        raise FileNotFoundError(args.experiments)
    experiments = pd.read_csv(args.experiments)
    missing_columns = REQUIRED_COLUMNS - set(experiments.columns)
    if missing_columns:
        raise ValueError(f"Bảng thí nghiệm thiếu các cột: {sorted(missing_columns)}")
    validation = experiments[experiments["split"] == "val"].copy()
    if validation.empty:
        raise ValueError("Chưa có kết quả xác thực để chọn mô hình")
    validation = validation.sort_values(args.metric, ascending=False)
    print(validation.to_string(index=False))
    print(f"\nMô hình đề xuất: {validation.iloc[0]['run_name']}")
    print("Tập test không được dùng trong bước xếp hạng này.")


if __name__ == "__main__":
    main()
