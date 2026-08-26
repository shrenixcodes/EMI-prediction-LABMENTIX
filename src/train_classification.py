"""
Train, evaluate, and MLflow-track classification models for EMI eligibility
prediction (Eligible / High_Risk / Not_Eligible).

Trains 4 models (Logistic Regression, Random Forest, XGBoost, Decision Tree),
logs params/metrics/artifacts/models to MLflow for each, selects the best
performer on the validation set, evaluates it on the held-out test set,
registers it in the MLflow Model Registry, and exports it (+ metadata) to
models/ for the Streamlit app.

Usage:
    python src/train_classification.py
"""
from __future__ import annotations

import json
import os
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from feature_engineering import FEATURE_COLUMNS, TARGET_CLASSIFICATION
from model_utils import (
    get_ohe_feature_names,
    load_splits,
    make_pipeline,
    save_bar_comparison,
    save_confusion_matrix,
    save_feature_importance,
    setup_mlflow,
)

MODELS_DIR = "models"
FIG_DIR = "reports/figures"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MODEL_FACTORY = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, solver="lbfgs"),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200, max_depth=18, min_samples_leaf=5, n_jobs=-1, random_state=42, class_weight="balanced"
    ),
    "XGBoost": lambda: XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
        objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1, random_state=42,
    ),
    "Decision Tree": lambda: DecisionTreeClassifier(max_depth=14, min_samples_leaf=10, random_state=42, class_weight="balanced"),
}


def evaluate(y_true, y_pred, y_proba, classes) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc_ovr_macro": roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr", labels=classes),
    }


def main():
    setup_mlflow("EMI_Eligibility_Classification")

    train, val, test = load_splits()
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train[TARGET_CLASSIFICATION])
    y_val = encoder.transform(val[TARGET_CLASSIFICATION])
    y_test = encoder.transform(test[TARGET_CLASSIFICATION])
    class_names = list(encoder.classes_)
    classes_encoded = list(range(len(class_names)))

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
            val_proba = pipe.predict_proba(X_val)
            metrics = evaluate(y_val, val_pred, val_proba, classes_encoded)
            metrics["train_time_sec"] = train_time

            print({k: round(v, 4) for k, v in metrics.items()})

            mlflow.log_param("model_type", name)
            mlflow.log_params({f"hp__{k}": v for k, v in estimator.get_params().items() if not callable(v)})
            mlflow.log_metrics(metrics)

            cm = confusion_matrix(y_val, val_pred)
            cm_path = os.path.join(FIG_DIR, f"cm_classification_{name.replace(' ', '_')}.png")
            save_confusion_matrix(cm, class_names, f"Confusion Matrix - {name} (Validation)", cm_path)
            mlflow.log_artifact(cm_path)

            feat_names = get_ohe_feature_names(pipe.named_steps["preprocess"])
            fi_path = os.path.join(FIG_DIR, f"fi_classification_{name.replace(' ', '_')}.png")
            if save_feature_importance(pipe.named_steps["model"], feat_names, f"Feature Importance - {name}", fi_path):
                mlflow.log_artifact(fi_path)

            mlflow.sklearn.log_model(pipe, name="model", input_example=X_train.head(3), serialization_format="cloudpickle")

            results[name] = metrics
            fitted_pipelines[name] = pipe

    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_pipe = fitted_pipelines[best_name]
    print(f"\n>>> Best classification model (by validation accuracy): {best_name}")

    test_pred = best_pipe.predict(X_test)
    test_proba = best_pipe.predict_proba(X_test)
    test_metrics = evaluate(y_test, test_pred, test_proba, classes_encoded)
    print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

    with mlflow.start_run(run_name=f"BEST__{best_name}__final_test_eval"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        cm_test = confusion_matrix(y_test, test_pred)
        cm_path = os.path.join(FIG_DIR, "cm_classification_BEST_test.png")
        save_confusion_matrix(cm_test, class_names, f"Confusion Matrix - {best_name} (TEST, held-out)", cm_path)
        mlflow.log_artifact(cm_path)
        model_info = mlflow.sklearn.log_model(best_pipe, name="model", input_example=X_train.head(3), serialization_format="cloudpickle")
        try:
            mlflow.register_model(model_info.model_uri, "emi_eligibility_classifier")
        except Exception as e:
            print(f"Model registry step skipped: {e}")

    comparison_path = os.path.join(FIG_DIR, "classification_model_comparison.png")
    save_bar_comparison(results, "accuracy", "Classification Model Comparison (Validation Accuracy)", comparison_path)

    joblib.dump(best_pipe, os.path.join(MODELS_DIR, "best_classification_model.pkl"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "classification_label_encoder.pkl"))

    metadata = {
        "best_model_name": best_name,
        "class_names": class_names,
        "feature_columns": FEATURE_COLUMNS,
        "validation_metrics": results,
        "test_metrics": test_metrics,
    }
    with open(os.path.join(MODELS_DIR, "classification_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    with open("reports/model_comparison_classification.md", "w", encoding="utf-8") as f:
        f.write("# Classification Model Comparison — EMI Eligibility\n\n")
        f.write("| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (OvR macro) | Train time (s) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, m in results.items():
            marker = " **(selected)**" if name == best_name else ""
            f.write(
                f"| {name}{marker} | {m['accuracy']:.4f} | {m['precision_macro']:.4f} | "
                f"{m['recall_macro']:.4f} | {m['f1_macro']:.4f} | {m['roc_auc_ovr_macro']:.4f} | "
                f"{m['train_time_sec']:.1f} |\n"
            )
        f.write(f"\n## Selected Model: {best_name}\n\n")
        f.write("Selected as the best validation-accuracy performer among all trained models.\n\n")
        f.write("### Held-out Test Set Performance\n\n")
        for k, v in test_metrics.items():
            f.write(f"- **{k}**: {v:.4f}\n")

    print("\nSaved best model, label encoder, metadata, and comparison report.")


if __name__ == "__main__":
    main()
