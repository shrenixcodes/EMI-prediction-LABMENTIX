"""
Train, evaluate, and MLflow-track regression models for maximum monthly EMI
amount prediction.

Trains 4 models (Linear Regression, Random Forest, XGBoost, Decision Tree),
logs params/metrics/artifacts/models to MLflow for each, selects the best
performer on the validation set (lowest RMSE), evaluates it on the held-out
test set, registers it in the MLflow Model Registry, and exports it (+
metadata) to models/ for the Streamlit app.

Usage:
    python src/train_regression.py
"""
from __future__ import annotations

import json
import os
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from feature_engineering import FEATURE_COLUMNS, TARGET_REGRESSION
from model_utils import (
    get_ohe_feature_names,
    load_splits,
    make_pipeline,
    save_bar_comparison,
    save_feature_importance,
    setup_mlflow,
)

MODELS_DIR = "models"
FIG_DIR = "reports/figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_FACTORY = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=42
    ),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=350, max_depth=6, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9,
        objective="reg:squarederror", n_jobs=-1, random_state=42,
    ),
    "Decision Tree": lambda: DecisionTreeRegressor(max_depth=14, min_samples_leaf=10, random_state=42),
}


def evaluate(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)) * 100,
    }


def save_pred_vs_actual(y_true, y_pred, title, path, n_sample=5000):
    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(n_sample, len(y_true)), replace=False)
    yt = np.asarray(y_true)[idx]
    yp = np.asarray(y_pred)[idx]
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(yt, yp, alpha=0.25, s=10, color="#1565C0")
    lims = [0, max(yt.max(), yp.max())]
    plt.plot(lims, lims, color="red", linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel("Actual max_monthly_emi")
    plt.ylabel("Predicted max_monthly_emi")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    setup_mlflow("EMI_MaxAmount_Regression")

    train, val, test = load_splits()
    y_train, y_val, y_test = train[TARGET_REGRESSION], val[TARGET_REGRESSION], test[TARGET_REGRESSION]
    X_train, X_val, X_test = train[FEATURE_COLUMNS], val[FEATURE_COLUMNS], test[FEATURE_COLUMNS]

    results = {}
    fitted_pipelines = {}

    for name, factory in MODEL_FACTORY.items():
        print(f"\n=== Training {name} ===")
        estimator = factory()
        pipe = make_pipeline(estimator)

        with mlflow.start_run(run_name=name):
            t0 = time.time()
            pipe.fit(X_train, y_train)
            train_time = time.time() - t0

            val_pred = pipe.predict(X_val)
            metrics = evaluate(y_val, val_pred)
            metrics["train_time_sec"] = train_time
            print({k: round(v, 4) for k, v in metrics.items()})

            mlflow.log_param("model_type", name)
            mlflow.log_params({f"hp__{k}": v for k, v in estimator.get_params().items() if not callable(v)})
            mlflow.log_metrics(metrics)

            scatter_path = os.path.join(FIG_DIR, f"scatter_regression_{name.replace(' ', '_')}.png")
            save_pred_vs_actual(y_val, val_pred, f"Predicted vs Actual - {name} (Validation)", scatter_path)
            mlflow.log_artifact(scatter_path)

            feat_names = get_ohe_feature_names(pipe.named_steps["preprocess"])
            fi_path = os.path.join(FIG_DIR, f"fi_regression_{name.replace(' ', '_')}.png")
            if save_feature_importance(pipe.named_steps["model"], feat_names, f"Feature Importance - {name}", fi_path):
                mlflow.log_artifact(fi_path)

            mlflow.sklearn.log_model(pipe, name="model", input_example=X_train.head(3), serialization_format="cloudpickle")

            results[name] = metrics
            fitted_pipelines[name] = pipe

    best_name = min(results, key=lambda n: results[n]["rmse"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\n>>> Best regression model (by validation RMSE): {best_name}")

    test_pred = best_pipe.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)
    print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

    with mlflow.start_run(run_name=f"BEST__{best_name}__final_test_eval"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        scatter_path = os.path.join(FIG_DIR, "scatter_regression_BEST_test.png")
        save_pred_vs_actual(y_test, test_pred, f"Predicted vs Actual - {best_name} (TEST, held-out)", scatter_path)
        mlflow.log_artifact(scatter_path)
        model_info = mlflow.sklearn.log_model(best_pipe, name="model", input_example=X_train.head(3), serialization_format="cloudpickle")
        try:
            mlflow.register_model(model_info.model_uri, "emi_max_amount_regressor")
        except Exception as e:
            print(f"Model registry step skipped: {e}")

    comparison_path = os.path.join(FIG_DIR, "regression_model_comparison.png")
    save_bar_comparison(results, "rmse", "Regression Model Comparison (Validation RMSE, lower=better)", comparison_path, higher_is_better=False)

    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "best_regression_model.pkl"))

    metadata = {
        "best_model_name": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "validation_metrics": results,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(MODELS_DIR, "regression_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open("reports/model_comparison_regression.md", "w", encoding="utf-8") as f:
        f.write("# Regression Model Comparison — Maximum Monthly EMI\n\n")
        f.write("| Model | RMSE (INR) | MAE (INR) | R² | MAPE (%) | Train time (s) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for name, m in results.items():
            marker = " **(selected)**" if name == best_name else ""
            f.write(
                f"| {name}{marker} | {m['rmse']:.2f} | {m['mae']:.2f} | {m['r2']:.4f} | "
                f"{m['mape']:.2f} | {m['train_time_sec']:.1f} |\n"
            )
        f.write(f"\n## Selected Model: {best_name}\n\n")
        f.write("Selected as the best validation-RMSE performer among all trained models.\n\n")
        f.write("### Held-out Test Set Performance\n\n")
        for k, v in test_metrics.items():
            f.write(f"- **{k}**: {v:.4f}\n")

    print("\nSaved best model, metadata, and comparison report.")


if __name__ == "__main__":
    main()
