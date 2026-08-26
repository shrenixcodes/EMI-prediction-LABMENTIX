"""
Data preprocessing pipeline for EMIPredict AI.

Loads the raw generated dataset, runs data-quality assessment, cleans
missing values / duplicates / invalid ranges, applies feature engineering,
and writes stratified train/validation/test splits to data/processed/.

Usage:
    python src/preprocessing.py --raw data/raw/emi_dataset.parquet
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from feature_engineering import (
    BASE_NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMNS,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    engineer_features,
)

VALID_RANGES = {
    "age": (18, 75),
    "monthly_salary": (0, 500_000),
    "credit_score": (300, 850),
    "family_size": (1, 15),
    "dependents": (0, 10),
    "requested_tenure": (1, 120),
}


def assess_quality(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated(subset=[c for c in df.columns if c not in ID_COLUMNS]).sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "missing_pct_overall": round(float(df.isna().mean().mean()) * 100, 4),
    }
    return report


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = len(df)

    # 1. Drop exact/near duplicates (ignore the synthetic ID column)
    df = df.drop_duplicates(subset=[c for c in df.columns if c not in ID_COLUMNS]).reset_index(drop=True)
    n_dupes_removed = before - len(df)

    # 2. Impute missing values
    numeric_missing = [c for c in BASE_NUMERIC_FEATURES if c in df.columns and df[c].isna().any()]
    for col in numeric_missing:
        df[col] = df[col].fillna(df[col].median())

    categorical_missing = [c for c in CATEGORICAL_FEATURES if c in df.columns and df[c].isna().any()]
    for col in categorical_missing:
        df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])

    # 3. Validate & clip out-of-range values rather than dropping records
    n_clipped = 0
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            mask = (df[col] < lo) | (df[col] > hi)
            n_clipped += int(mask.sum())
            df[col] = df[col].clip(lo, hi)

    # 4. Drop rows with a non-positive salary (cannot compute financial ratios)
    n_before_salary_filter = len(df)
    df = df[df["monthly_salary"] > 0].reset_index(drop=True)
    n_invalid_salary_removed = n_before_salary_filter - len(df)

    cleaning_report = {
        "duplicate_rows_removed": int(n_dupes_removed),
        "numeric_columns_imputed": numeric_missing,
        "categorical_columns_imputed": categorical_missing,
        "out_of_range_values_clipped": int(n_clipped),
        "invalid_salary_rows_removed": int(n_invalid_salary_removed),
        "final_row_count": int(len(df)),
    }
    return df, cleaning_report


def split_and_save(df: pd.DataFrame, out_dir: str, seed: int = 42) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df[TARGET_CLASSIFICATION]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df[TARGET_CLASSIFICATION]
    )

    train_df.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(out_dir, "val.parquet"), index=False)
    test_df.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)

    return {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
    }


def main():
    parser = argparse.ArgumentParser(description="Clean and split the EMIPredict AI dataset")
    parser.add_argument("--raw", type=str, default="data/raw/emi_dataset.parquet")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--report", type=str, default="reports/data_quality_report.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading raw dataset from {args.raw} ...")
    df = pd.read_parquet(args.raw) if args.raw.endswith(".parquet") else pd.read_csv(args.raw)
    print(f"Raw shape: {df.shape}")

    quality_before = assess_quality(df)
    print("Quality assessment (before cleaning):")
    print(json.dumps(quality_before, indent=2))

    df_clean, cleaning_report = clean(df)
    print("\nCleaning report:")
    print(json.dumps(cleaning_report, indent=2))

    quality_after = assess_quality(df_clean)

    print("\nApplying feature engineering ...")
    df_engineered = engineer_features(df_clean)

    print("Splitting into train/val/test (70/15/15, stratified on eligibility) ...")
    split_stats = split_and_save(df_engineered, args.out_dir, seed=args.seed)
    print(json.dumps(split_stats, indent=2))

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    full_report = {
        "quality_before_cleaning": quality_before,
        "cleaning_actions": cleaning_report,
        "quality_after_cleaning": quality_after,
        "split_stats": split_stats,
        "classification_target": TARGET_CLASSIFICATION,
        "regression_target": TARGET_REGRESSION,
        "class_balance": df_engineered[TARGET_CLASSIFICATION].value_counts(normalize=True).round(4).to_dict(),
    }
    with open(args.report, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nSaved data quality report to {args.report}")


if __name__ == "__main__":
    main()
