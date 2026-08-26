"""Shared utilities for the EMIPredict AI Streamlit app: model loading,
prediction helpers, and a SQLite-backed CRUD layer for loan applications.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "app_data.db")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from feature_engineering import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    engineer_features,
)

EMI_SCENARIOS = [
    "E-commerce Shopping EMI",
    "Home Appliances EMI",
    "Vehicle EMI",
    "Personal Loan EMI",
    "Education EMI",
]

FORM_FIELDS = [
    "age", "gender", "marital_status", "education", "monthly_salary", "employment_type",
    "years_of_employment", "company_type", "house_type", "monthly_rent", "family_size",
    "dependents", "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
    "other_monthly_expenses", "existing_loans", "current_emi_amount", "credit_score",
    "bank_balance", "emergency_fund", "emi_scenario", "requested_amount", "requested_tenure",
]


# --------------------------------------------------------------------------- #
# Model loading (cached across reruns within a session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_classification_artifacts():
    model_path = os.path.join(MODELS_DIR, "best_classification_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, "classification_label_encoder.pkl")
    meta_path = os.path.join(MODELS_DIR, "classification_metadata.json")
    if not os.path.exists(model_path):
        return None, None, None
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, encoder, metadata


@st.cache_resource(show_spinner=False)
def load_regression_artifacts():
    model_path = os.path.join(MODELS_DIR, "best_regression_model.pkl")
    meta_path = os.path.join(MODELS_DIR, "regression_metadata.json")
    if not os.path.exists(model_path):
        return None, None
    model = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, metadata


def models_available() -> bool:
    return os.path.exists(os.path.join(MODELS_DIR, "best_classification_model.pkl")) and os.path.exists(
        os.path.join(MODELS_DIR, "best_regression_model.pkl")
    )


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def build_feature_row(raw_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])
    engineered = engineer_features(df)
    return engineered[FEATURE_COLUMNS]


def predict_eligibility(raw_input: dict):
    model, encoder, metadata = load_classification_artifacts()
    if model is None:
        return None
    X = build_feature_row(raw_input)
    pred_encoded = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    label = encoder.inverse_transform([pred_encoded])[0]
    class_probs = dict(zip(encoder.classes_, proba.round(4)))
    return {
        "label": label,
        "confidence": float(proba.max()),
        "class_probabilities": class_probs,
        "model_name": metadata["best_model_name"],
    }


def predict_max_emi(raw_input: dict):
    model, metadata = load_regression_artifacts()
    if model is None:
        return None
    X = build_feature_row(raw_input)
    pred = float(model.predict(X)[0])
    return {"predicted_max_emi": round(pred, 2), "model_name": metadata["best_model_name"]}


# --------------------------------------------------------------------------- #
# CRUD layer (SQLite) — "financial data management"
# --------------------------------------------------------------------------- #
def get_connection() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS loan_applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT,
            age INTEGER, gender TEXT, marital_status TEXT, education TEXT,
            monthly_salary REAL, employment_type TEXT, years_of_employment REAL, company_type TEXT,
            house_type TEXT, monthly_rent REAL, family_size INTEGER, dependents INTEGER,
            school_fees REAL, college_fees REAL, travel_expenses REAL, groceries_utilities REAL,
            other_monthly_expenses REAL, existing_loans TEXT, current_emi_amount REAL,
            credit_score INTEGER, bank_balance REAL, emergency_fund REAL,
            emi_scenario TEXT, requested_amount REAL, requested_tenure INTEGER,
            predicted_eligibility TEXT, eligibility_confidence REAL, predicted_max_emi REAL,
            created_at TEXT, updated_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def create_application(record: dict) -> int:
    init_db()
    conn = get_connection()
    now = datetime.now(timezone.utc).isoformat()
    record = {**record, "created_at": now, "updated_at": now}
    cols = ", ".join(record.keys())
    placeholders = ", ".join(["?"] * len(record))
    cur = conn.execute(
        f"INSERT INTO loan_applications ({cols}) VALUES ({placeholders})", list(record.values())
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return new_id


def read_applications(search: str | None = None) -> pd.DataFrame:
    init_db()
    conn = get_connection()
    query = "SELECT * FROM loan_applications"
    params: list = []
    if search:
        query += " WHERE customer_name LIKE ? OR emi_scenario LIKE ? OR predicted_eligibility LIKE ?"
        like = f"%{search}%"
        params = [like, like, like]
    query += " ORDER BY id DESC"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def get_application(record_id: int) -> dict | None:
    init_db()
    conn = get_connection()
    row = conn.execute("SELECT * FROM loan_applications WHERE id = ?", (record_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_application(record_id: int, updates: dict):
    init_db()
    conn = get_connection()
    updates = {**updates, "updated_at": datetime.now(timezone.utc).isoformat()}
    set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
    conn.execute(
        f"UPDATE loan_applications SET {set_clause} WHERE id = ?", list(updates.values()) + [record_id]
    )
    conn.commit()
    conn.close()


def delete_application(record_id: int):
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM loan_applications WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()


def delete_all_applications():
    init_db()
    conn = get_connection()
    conn.execute("DELETE FROM loan_applications")
    conn.commit()
    conn.close()
