"""
Synthetic dataset generator for EMIPredict AI.

Generates 400,000 realistic financial-profile records across 5 EMI lending
scenarios (80,000 each), covering the 22+ demographic/financial variables
described in the project brief, plus the two ML targets:

    - emi_eligibility (classification): Eligible / High_Risk / Not_Eligible
    - max_monthly_emi (regression): maximum safe monthly EMI in INR

The targets are derived from an underlying "financial capacity" simulation
(disposable income, affordability ratio driven by credit score, employment
stability, dependents, emergency savings, etc.) plus injected label noise,
so the relationship is realistic and learnable but not trivially separable.

Usage:
    python src/generate_dataset.py --n-per-scenario 80000 --out data/raw/emi_dataset.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

RNG_SEED = 42

SCENARIOS = {
    "E-commerce Shopping EMI": {
        "amount": (10_000, 200_000),
        "tenure": (3, 24),
        "annual_rate": (0.14, 0.22),
    },
    "Home Appliances EMI": {
        "amount": (20_000, 300_000),
        "tenure": (6, 36),
        "annual_rate": (0.12, 0.18),
    },
    "Vehicle EMI": {
        "amount": (80_000, 1_500_000),
        "tenure": (12, 84),
        "annual_rate": (0.08, 0.13),
    },
    "Personal Loan EMI": {
        "amount": (50_000, 1_000_000),
        "tenure": (12, 60),
        "annual_rate": (0.11, 0.18),
    },
    "Education EMI": {
        "amount": (50_000, 500_000),
        "tenure": (6, 48),
        "annual_rate": (0.09, 0.14),
    },
}

EDUCATION_LEVELS = ["High School", "Graduate", "Post Graduate", "Professional"]
EDUCATION_WEIGHTS = [0.20, 0.40, 0.28, 0.12]

EMPLOYMENT_TYPES = ["Private", "Government", "Self-employed"]
EMPLOYMENT_WEIGHTS = [0.60, 0.20, 0.20]

COMPANY_TYPES = ["Startup", "SME", "Large Corporate", "MNC", "Government", "Not Applicable"]

HOUSE_TYPES = ["Rented", "Own", "Family"]


def emi_formula(principal: np.ndarray, annual_rate: np.ndarray, tenure_months: np.ndarray) -> np.ndarray:
    r = annual_rate / 12.0
    n = tenure_months
    factor = (1 + r) ** n
    # avoid division by zero when r == 0
    r_safe = np.where(r == 0, 1e-9, r)
    emi = principal * r_safe * factor / (factor - 1)
    return emi


def generate(n_per_scenario: int, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_per_scenario * len(SCENARIOS)

    # ---- Scenario assignment -------------------------------------------------
    scenario_names = list(SCENARIOS.keys())
    emi_scenario = np.repeat(scenario_names, n_per_scenario)
    rng.shuffle(emi_scenario)

    # ---- Personal demographics -------------------------------------------------
    age = np.clip(rng.normal(38, 9, n), 25, 60).round().astype(int)
    gender = rng.choice(["Male", "Female"], size=n, p=[0.56, 0.44])
    marital_prob = np.clip((age - 25) / 35 * 0.7 + 0.15, 0.15, 0.85)
    marital_status = np.where(rng.random(n) < marital_prob, "Married", "Single")
    education = rng.choice(EDUCATION_LEVELS, size=n, p=EDUCATION_WEIGHTS)
    edu_income_multiplier = pd.Series(education).map(
        {"High School": 0.75, "Graduate": 1.0, "Post Graduate": 1.3, "Professional": 1.7}
    ).to_numpy()

    # ---- Employment & income -----------------------------------------------
    employment_type = rng.choice(EMPLOYMENT_TYPES, size=n, p=EMPLOYMENT_WEIGHTS)
    emp_stability = pd.Series(employment_type).map(
        {"Government": 1.15, "Private": 1.0, "Self-employed": 0.85}
    ).to_numpy()

    base_salary = rng.lognormal(mean=10.7, sigma=0.45, size=n)
    monthly_salary = np.clip(base_salary * edu_income_multiplier * emp_stability, 15_000, 200_000)

    max_possible_experience = np.maximum(age - 22, 0)
    years_of_employment = np.clip(
        rng.gamma(shape=2.2, scale=3.0, size=n), 0, max_possible_experience
    ).round(1)

    company_type = np.where(
        employment_type == "Self-employed",
        "Not Applicable",
        rng.choice(["Startup", "SME", "Large Corporate", "MNC", "Government"], size=n),
    )
    # Government employment_type should map to Government company_type
    company_type = np.where(employment_type == "Government", "Government", company_type)

    # ---- Housing & family ----------------------------------------------------
    own_prob = np.clip((age - 25) / 35 * 0.5 + 0.15 + (monthly_salary / 200_000) * 0.2, 0.05, 0.75)
    house_roll = rng.random(n)
    house_type = np.select(
        [house_roll < own_prob, house_roll < own_prob + 0.3],
        ["Own", "Family"],
        default="Rented",
    )
    monthly_rent = np.where(
        house_type == "Rented",
        np.clip(monthly_salary * rng.uniform(0.15, 0.35, n), 3_000, 60_000),
        0.0,
    ).round(2)

    family_size = np.clip(rng.poisson(3.2, n) + 1, 1, 9)
    dependents = np.clip(
        (family_size - 1 - (marital_status == "Single").astype(int)).clip(min=0)
        - rng.integers(0, 2, n),
        0,
        6,
    )

    # ---- Monthly financial obligations ---------------------------------------
    n_school_children = np.clip(dependents, 0, 3) * (age < 50)
    school_fees = np.where(
        n_school_children > 0, n_school_children * rng.uniform(1_500, 6_000, n), 0.0
    ).round(2)

    has_college_dependent = (dependents > 1) & (age > 40) & (rng.random(n) < 0.35)
    college_fees = np.where(
        has_college_dependent, rng.uniform(5_000, 25_000, n), 0.0
    ).round(2)

    travel_expenses = np.clip(monthly_salary * rng.uniform(0.03, 0.12, n), 500, 15_000).round(2)
    groceries_utilities = np.clip(
        (family_size * rng.uniform(1_800, 3_500, n)) + monthly_salary * 0.02, 2_000, 40_000
    ).round(2)
    other_monthly_expenses = np.clip(monthly_salary * rng.uniform(0.02, 0.10, n), 500, 20_000).round(2)

    # ---- Financial status & credit history ------------------------------------
    existing_loan_prob = np.clip(0.25 + (age - 25) / 70, 0.1, 0.55)
    existing_loans = np.where(rng.random(n) < existing_loan_prob, "Yes", "No")
    current_emi_amount = np.where(
        existing_loans == "Yes",
        np.clip(monthly_salary * rng.uniform(0.05, 0.30, n), 500, 40_000),
        0.0,
    ).round(2)

    credit_base = (
        500
        + (monthly_salary / 200_000) * 150
        + (years_of_employment / 30) * 80
        + (employment_type == "Government") * 40
        - (existing_loans == "Yes") * 30
        + rng.normal(0, 60, n)
    )
    credit_score = np.clip(credit_base, 300, 850).round().astype(int)

    savings_capacity = np.clip(monthly_salary - monthly_rent - current_emi_amount, 0, None)
    bank_balance = np.clip(
        savings_capacity * rng.uniform(1.0, 8.0, n) * (years_of_employment / 5 + 0.3),
        1_000,
        2_000_000,
    ).round(2)
    emergency_fund = np.clip(bank_balance * rng.uniform(0.1, 0.6, n), 0, 1_000_000).round(2)

    # ---- Loan application details ---------------------------------------------
    amount_lo = np.array([SCENARIOS[s]["amount"][0] for s in emi_scenario], dtype=float)
    amount_hi = np.array([SCENARIOS[s]["amount"][1] for s in emi_scenario], dtype=float)
    tenure_lo = np.array([SCENARIOS[s]["tenure"][0] for s in emi_scenario], dtype=float)
    tenure_hi = np.array([SCENARIOS[s]["tenure"][1] for s in emi_scenario], dtype=float)
    rate_lo = np.array([SCENARIOS[s]["annual_rate"][0] for s in emi_scenario])
    rate_hi = np.array([SCENARIOS[s]["annual_rate"][1] for s in emi_scenario])

    # income-skewed position within the scenario's amount range: most applicants
    # request modest amounts relative to their means (beta skewed low), with a
    # meaningful pull toward higher income -> higher requested amount
    income_pct = np.clip((monthly_salary - 15_000) / (200_000 - 15_000), 0, 1)
    amount_pos = np.clip(rng.beta(1.6, 3.2, n) * 0.45 + income_pct * 0.55, 0, 1)
    requested_amount = (amount_lo + amount_pos * (amount_hi - amount_lo)).round(2)

    tenure_pos = np.clip(amount_pos * 0.5 + rng.random(n) * 0.5, 0, 1)
    requested_tenure = (tenure_lo + tenure_pos * (tenure_hi - tenure_lo)).round().astype(int)

    annual_rate = rng.uniform(rate_lo, rate_hi)
    requested_emi = emi_formula(requested_amount, annual_rate, requested_tenure).round(2)

    # ---- Target derivation: financial-capacity simulation ----------------------
    disposable_income = np.clip(
        monthly_salary
        - monthly_rent
        - school_fees
        - college_fees
        - travel_expenses
        - groceries_utilities
        - other_monthly_expenses
        - current_emi_amount,
        0,
        None,
    )

    affordability_ratio = (
        0.40
        + (credit_score - 300) / 550 * 0.22
        + (employment_type == "Government") * 0.05
        + (employment_type == "Self-employed") * -0.05
        + np.clip(years_of_employment / 20, 0, 0.10)
        - dependents * 0.015
        + (emergency_fund > 3 * monthly_salary) * 0.03
    )
    affordability_ratio = np.clip(affordability_ratio, 0.20, 0.75)

    noise = rng.normal(1.0, 0.06, n)
    max_monthly_emi = np.clip(disposable_income * affordability_ratio * noise, 500, 50_000).round(2)

    eps = 1e-6
    burden_ratio = requested_emi / (max_monthly_emi + eps)

    eligibility = np.select(
        [
            (burden_ratio <= 0.95) & (credit_score >= 600),
            (burden_ratio <= 1.35) & (credit_score >= 500),
        ],
        ["Eligible", "High_Risk"],
        default="Not_Eligible",
    )
    # inject small label noise to keep the problem realistic (non-trivial)
    flip_mask = rng.random(n) < 0.04
    flip_choices = rng.choice(["Eligible", "High_Risk", "Not_Eligible"], size=n)
    emi_eligibility = np.where(flip_mask, flip_choices, eligibility)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST{100000+i}" for i in range(n)],
            "age": age,
            "gender": gender,
            "marital_status": marital_status,
            "education": education,
            "monthly_salary": monthly_salary.round(2),
            "employment_type": employment_type,
            "years_of_employment": years_of_employment,
            "company_type": company_type,
            "house_type": house_type,
            "monthly_rent": monthly_rent,
            "family_size": family_size,
            "dependents": dependents,
            "school_fees": school_fees,
            "college_fees": college_fees,
            "travel_expenses": travel_expenses,
            "groceries_utilities": groceries_utilities,
            "other_monthly_expenses": other_monthly_expenses,
            "existing_loans": existing_loans,
            "current_emi_amount": current_emi_amount,
            "credit_score": credit_score,
            "bank_balance": bank_balance,
            "emergency_fund": emergency_fund,
            "emi_scenario": emi_scenario,
            "requested_amount": requested_amount,
            "requested_tenure": requested_tenure,
            "emi_eligibility": emi_eligibility,
            "max_monthly_emi": max_monthly_emi,
        }
    )

    # small amount of realistic missingness / duplicates for the preprocessing
    # step to demonstrate data-quality handling
    missing_cols = ["years_of_employment", "emergency_fund", "bank_balance", "company_type"]
    for col in missing_cols:
        mask = rng.random(n) < 0.004
        df.loc[mask, col] = np.nan

    dup_idx = rng.choice(df.index, size=int(n * 0.001), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Generate the EMIPredict AI synthetic dataset")
    parser.add_argument("--n-per-scenario", type=int, default=80_000)
    parser.add_argument("--out", type=str, default="data/raw/emi_dataset.parquet")
    parser.add_argument("--sample-csv", type=str, default="data/raw/emi_dataset_sample.csv")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    print(f"Generating {args.n_per_scenario * len(SCENARIOS):,} records across {len(SCENARIOS)} scenarios...")
    df = generate(args.n_per_scenario, seed=args.seed)
    print(f"Generated shape: {df.shape}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.out.endswith(".parquet"):
        df.to_parquet(args.out, index=False)
    else:
        df.to_csv(args.out, index=False)
    print(f"Saved full dataset to {args.out}")

    sample = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed)
    os.makedirs(os.path.dirname(args.sample_csv), exist_ok=True)
    sample.to_csv(args.sample_csv, index=False)
    print(f"Saved {len(sample):,}-row sample CSV to {args.sample_csv}")

    print("\nClass balance (emi_eligibility):")
    print(df["emi_eligibility"].value_counts(normalize=True).round(3))
    print("\nmax_monthly_emi describe():")
    print(df["max_monthly_emi"].describe().round(2))


if __name__ == "__main__":
    main()
