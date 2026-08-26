import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ROOT_DIR, load_classification_artifacts, load_regression_artifacts  # noqa: E402

st.set_page_config(page_title="Model Performance | EMIPredict AI", page_icon="\U0001F4C8", layout="wide")
st.title("\U0001F4C8 Model Performance & MLflow Experiment Tracking")

FIG_DIR = os.path.join(ROOT_DIR, "reports", "figures")
MLFLOW_DB = os.path.join(ROOT_DIR, "mlflow.db")

clf_model, clf_encoder, clf_meta = load_classification_artifacts()
reg_model, reg_meta = load_regression_artifacts()

if clf_meta is None or reg_meta is None:
    st.warning("Run the training scripts to populate model metadata: "
               "`python src/train_classification.py && python src/train_regression.py`")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Classification Models", "Regression Models", "MLflow Runs"])

with tab1:
    st.markdown("### EMI Eligibility — Classification Model Comparison")
    df = pd.DataFrame(clf_meta["validation_metrics"]).T
    df.index.name = "Model"
    st.dataframe(
        df.style.format("{:.4f}").highlight_max(subset=["accuracy", "f1_macro", "roc_auc_ovr_macro"], color="#c8e6c9"),
        use_container_width=True,
    )
    st.success(f"**Selected for deployment:** {clf_meta['best_model_name']}")

    img_path = os.path.join(FIG_DIR, "classification_model_comparison.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Validation accuracy across all trained classification models")

    st.markdown("#### Held-out Test Set Performance (selected model)")
    test_m = clf_meta["test_metrics"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{test_m['accuracy']:.2%}")
    c2.metric("Precision (macro)", f"{test_m['precision_macro']:.2%}")
    c3.metric("Recall (macro)", f"{test_m['recall_macro']:.2%}")
    c4.metric("F1 (macro)", f"{test_m['f1_macro']:.2%}")
    c5.metric("ROC-AUC (OvR)", f"{test_m['roc_auc_ovr_macro']:.3f}")

    cm_path = os.path.join(FIG_DIR, "cm_classification_BEST_test.png")
    if os.path.exists(cm_path):
        col1, col2 = st.columns(2)
        col1.image(cm_path, caption="Confusion matrix (test set)")
        fi_name = f"fi_classification_{clf_meta['best_model_name'].replace(' ', '_')}.png"
        fi_path = os.path.join(FIG_DIR, fi_name)
        if os.path.exists(fi_path):
            col2.image(fi_path, caption="Feature importance")

with tab2:
    st.markdown("### Maximum Monthly EMI — Regression Model Comparison")
    df2 = pd.DataFrame(reg_meta["validation_metrics"]).T
    df2.index.name = "Model"
    st.dataframe(
        df2.style.format("{:.4f}").highlight_min(subset=["rmse", "mae", "mape"], color="#c8e6c9")
        .highlight_max(subset=["r2"], color="#c8e6c9"),
        use_container_width=True,
    )
    st.success(f"**Selected for deployment:** {reg_meta['best_model_name']}")

    img_path = os.path.join(FIG_DIR, "regression_model_comparison.png")
    if os.path.exists(img_path):
        st.image(img_path, caption="Validation RMSE across all trained regression models (lower is better)")

    st.markdown("#### Held-out Test Set Performance (selected model)")
    test_m2 = reg_meta["test_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"₹{test_m2['rmse']:,.0f}")
    c2.metric("MAE", f"₹{test_m2['mae']:,.0f}")
    c3.metric("R²", f"{test_m2['r2']:.4f}")
    c4.metric("MAPE", f"{test_m2['mape']:.2f}%")

    scatter_path = os.path.join(FIG_DIR, "scatter_regression_BEST_test.png")
    if os.path.exists(scatter_path):
        col1, col2 = st.columns(2)
        col1.image(scatter_path, caption="Predicted vs Actual (test set)")
        fi_name = f"fi_regression_{reg_meta['best_model_name'].replace(' ', '_')}.png"
        fi_path = os.path.join(FIG_DIR, fi_name)
        if os.path.exists(fi_path):
            col2.image(fi_path, caption="Feature importance")

with tab3:
    st.markdown("### MLflow Experiment Tracking")
    st.caption(
        "All runs below (params, metrics, and model/confusion-matrix artifacts) were logged to a local "
        "MLflow tracking server (`sqlite:///mlflow.db`). Launch the full MLflow UI to browse interactively:"
    )
    st.code("mlflow ui --backend-store-uri sqlite:///mlflow.db", language="bash")

    if os.path.exists(MLFLOW_DB):
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
            client = MlflowClient()
            experiments = client.search_experiments()
            for exp in experiments:
                if exp.name == "Default":
                    continue
                runs = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=25)
                if not runs:
                    continue
                st.markdown(f"#### Experiment: `{exp.name}`  ·  {len(runs)} run(s)")
                rows = []
                for r in runs:
                    row = {"run_name": r.data.tags.get("mlflow.runName", r.info.run_id[:8]), "status": r.info.status}
                    row.update(r.data.metrics)
                    rows.append(row)
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        except Exception as e:
            st.info(f"MLflow tracking DB present but could not be queried in this environment: {e}")
    else:
        st.info("No local `mlflow.db` found in this deployment. Metrics are still available via the "
                "tables/plots above, which are exported directly from the training run.")
