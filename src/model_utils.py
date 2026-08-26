"""Shared helpers for model training scripts: preprocessing pipeline builder,
MLflow experiment setup, and plotting utilities.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MLFLOW_DB_PATH = os.path.abspath("mlflow.db")


def setup_mlflow(experiment_name: str):
    # The legacy filesystem tracking store is in maintenance mode in MLflow 3.x;
    # a local SQLite backend gives full tracking + registry support.
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.replace(os.sep, '/')}")
    mlflow.set_experiment(experiment_name)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])


def load_splits():
    train = pd.read_parquet("data/processed/train.parquet")
    val = pd.read_parquet("data/processed/val.parquet")
    test = pd.read_parquet("data/processed/test.parquet")
    return train, val, test


def save_confusion_matrix(cm, class_names, title, path):
    import numpy as np
    import seaborn as sns

    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_bar_comparison(results: dict, metric: str, title: str, path: str, higher_is_better: bool = True):
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    colors = ["#2E7D32" if v == (max(values) if higher_is_better else min(values)) else "#90A4AE" for v in values]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(names, values, color=colors)
    plt.title(title)
    plt.ylabel(metric)
    plt.xticks(rotation=15, ha="right")
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_feature_importance(model, feature_names, title, path, top_n=15):
    import numpy as np

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = abs(coef).mean(axis=0) if coef.ndim > 1 else abs(coef)
    else:
        return None

    order = list(reversed(sorted(range(len(importances)), key=lambda i: importances[i])))[:top_n]
    names = [feature_names[i] for i in order]
    vals = [importances[i] for i in order]

    plt.figure(figsize=(7, 6))
    plt.barh(names[::-1], vals[::-1], color="#1565C0")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def get_ohe_feature_names(preprocessor: ColumnTransformer):
    cat_names = list(preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    return cat_names + list(NUMERIC_FEATURES)
