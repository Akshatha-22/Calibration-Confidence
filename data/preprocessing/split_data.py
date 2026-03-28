"""Create train/val/test splits for FinSen classification."""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_col: str = "Category",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train/val/test sets with optional stratification."""
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Proportions must sum to 1.0")

    stratify = df[stratify_col] if stratify_col in df.columns else None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify,
    )

    stratify_temp = temp_df[stratify_col] if stratify_col in temp_df.columns else None
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size / (val_size + test_size),
        random_state=random_state,
        stratify=stratify_temp,
    )

    return train_df, val_df, test_df


def create_splits(
    input_csv: str = "data/finsen/raw/FinSen_US_Categorized.csv",
    output_dir: str = "data/finsen/processed",
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    stratify_col: str = "Category",
) -> Tuple[str, str, str]:
    """Create and save train/val/test CSV splits."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    if stratify_col not in df.columns:
        raise ValueError(
            f"Expected '{stratify_col}' column in {input_csv}. "
            "Use a categorized FinSen CSV with labels."
        )

    train_df, val_df, test_df = split_dataset(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify_col=stratify_col,
    )

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/val/test splits for FinSen classification")
    parser.add_argument("--input-csv", type=str, default="data/finsen/raw/FinSen_US_Categorized.csv")
    parser.add_argument("--output-dir", type=str, default="data/finsen/processed")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--stratify-col", type=str, default="Category")
    args = parser.parse_args()

    train_path, val_path, test_path = create_splits(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify_col=args.stratify_col,
    )

    print("Splits saved:")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")
    print(f"  test:  {test_path}")


if __name__ == "__main__":
    main()
