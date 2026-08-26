"""
Feature engineering for EMIPredict AI.

Shared between the training pipeline and the Streamlit app so that a single
customer record entered in the UI is transformed with exactly the same logic
used during model training (no train/serve skew).

`engineer_features` only *adds derived columns* — categorical encoding and
numeric scaling are handled inside the sklearn model Pipelines themselves
(see src/train_classification.py / src/train_regression.py), so this module
has no fitted state to persist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Average annual interest rate assumed per EMI scenario (used to derive the
# requested EMI from requested_amount / requested_tenure, mirroring how an
# underwriting system would apply a standard rate card).
SCENARIO_RATE_TABLE = {
    "E-commerce Shopping EMI": 0.18,
    "Home Appliances EMI": 0.15,
    "Vehicle EMI": 0.105,
    "Personal Loan EMI": 0.145,
    "Education EMI": 0.115,
}

ID_COLUMNS = ["customer_id"]
TARGET_CLASSIFICATION = "emi_eligibility"
TARGET_REGRESSION = "max_monthly_emi"

CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "education",
    "employment_type",
    "company_type",
    "house_type",
    "existing_loans",
    "emi_scenario",
]

BASE_NUMERIC_FEATURES = [
    "age",
    "monthly_salary",
    "years_of_employment",
    "monthly_rent",
    "family_size",
    "dependents",
    "school_fees",
    "college_fees",
    "travel_expenses",
    "groceries_utilities",
    "other_monthly_expenses",
    "current_emi_amount",
    "credit_score",
    "bank_balance",
    "emergency_fund",
    "requested_amount",
    "requested_tenure",
]

ENGINEERED_NUMERIC_FEATURES = [
    "requested_emi",
    "total_monthly_expenses",
    "total_monthly_obligations",
    "disposable_income",
    "debt_to_income_ratio",
    "expense_to_income_ratio",
    "affordability_ratio",
    "requested_to_income_ratio",
    "requested_burden_ratio",
    "employment_stability_score",
    "income_per_dependent",
    "emergency_fund_months",
    "savings_to_income_ratio",
    "credit_score_norm",
]

NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _emi_formula(principal: pd.Series, annual_rate: pd.Series, tenure_months: pd.Series) -> pd.Series:
    r = annual_rate / 12.0
    n = tenure_months
    factor = (1 + r) ** n
    r_safe = r.where(r != 0, 1e-9)
    return principal * r_safe * factor / (factor - 1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived financial-ratio and risk-scoring features to a raw dataframe.

    Expects the 22+ raw input columns described in the project brief. Safe to
    call on a single-row dataframe (Streamlit form submission) or the full
    400K-row dataset.
    """
    df = df.copy()

    annual_rate = df["emi_scenario"].map(SCENARIO_RATE_TABLE).fillna(0.14)
    df["requested_emi"] = _emi_formula(
        df["requested_amount"], annual_rate, df["requested_tenure"]
    ).round(2)

    expense_cols = [
        "school_fees",
        "college_fees",
        "travel_expenses",
        "groceries_utilities",
        "other_monthly_expenses",
    ]
    df["total_monthly_expenses"] = df[expense_cols].sum(axis=1)
    df["total_monthly_obligations"] = (
        df["total_monthly_expenses"] + df["monthly_rent"] + df["current_emi_amount"]
    )
    df["disposable_income"] = (df["monthly_salary"] - df["total_monthly_obligations"]).clip(lower=0)

    salary_safe = df["monthly_salary"].replace(0, np.nan)
    df["debt_to_income_ratio"] = (
        (df["current_emi_amount"] + df["monthly_rent"]) / salary_safe
    ).fillna(0).round(4)
    df["expense_to_income_ratio"] = (df["total_monthly_expenses"] / salary_safe).fillna(0).round(4)
    df["affordability_ratio"] = (df["disposable_income"] / salary_safe).fillna(0).round(4)
    df["requested_to_income_ratio"] = (df["requested_emi"] / salary_safe).fillna(0).round(4)

    disposable_safe = df["disposable_income"].replace(0, np.nan)
    df["requested_burden_ratio"] = (
        (df["requested_emi"] / disposable_safe).fillna(df["requested_emi"]).clip(upper=10).round(4)
    )

    experience_capacity = (df["age"] - 22).clip(lower=1)
    df["employment_stability_score"] = (
        (df["years_of_employment"] / experience_capacity).clip(upper=1).round(4)
    )

    df["income_per_dependent"] = (df["monthly_salary"] / (df["dependents"] + 1)).round(2)
    df["emergency_fund_months"] = (df["emergency_fund"] / salary_safe).fillna(0).round(4)
    df["savings_to_income_ratio"] = (df["bank_balance"] / (salary_safe * 12)).fillna(0).round(4)
    df["credit_score_norm"] = ((df["credit_score"] - 300) / (850 - 300)).round(4)

    return df


def select_model_frame(df: pd.DataFrame, target: str | None = None) -> pd.DataFrame:
    """Return only the columns the models are trained on (+ optional target)."""
    cols = FEATURE_COLUMNS + ([target] if target else [])
    return df[cols]
