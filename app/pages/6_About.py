import streamlit as st

st.set_page_config(page_title="About | EMIPredict AI", page_icon="ℹ️", layout="wide")
st.title("ℹ️ About EMIPredict AI")

st.markdown(
    """
### Problem Statement
People frequently struggle to repay EMIs due to poor financial planning and
inadequate risk assessment at the point of lending. **EMIPredict AI** tackles
this with a data-driven platform that gives both applicants and lenders a
clear, real-time view of affordability and risk *before* a loan is issued.

### Architecture
```
Dataset (400K records)
      ↓
Data Quality Assessment & Preprocessing   (src/preprocessing.py)
      ↓
Feature Engineering & EDA                 (src/feature_engineering.py, src/eda.py)
      ↓
ML Model Training & MLflow Tracking       (src/train_classification.py, src/train_regression.py)
      ↓
Model Evaluation & Selection              (MLflow experiment comparison → best model)
      ↓
Streamlit Application                     (app/)
      ↓
Cloud Deployment (Streamlit Cloud)
```

### Business Use Cases
- **Financial Institutions** — automate loan approval, cut manual underwriting time, real-time eligibility checks for walk-ins.
- **FinTech Companies** — instant EMI eligibility for digital lending, mobile pre-qualification, automated risk scoring.
- **Banks & Credit Agencies** — data-driven loan amount recommendations, portfolio risk management, documented compliance trail.
- **Loan Officers & Underwriters** — AI-assisted approval recommendations, full financial profile analysis in seconds.

### Tech Stack
Python · pandas · scikit-learn · XGBoost · MLflow · Streamlit · Plotly · SQLite

### Models
- **Classification (EMI eligibility):** Logistic Regression, Random Forest, XGBoost, Decision Tree
- **Regression (max monthly EMI):** Linear Regression, Random Forest, XGBoost, Decision Tree

All 8 models are trained with full MLflow experiment tracking (params, metrics,
confusion matrices / prediction plots, feature importance, and the model
artifact itself), and the best performer in each task is registered to the
MLflow Model Registry and served by this app.
"""
)

st.divider()
st.caption("EMIPredict AI · FinTech & Banking Capstone Project · Domain: FinTech and Banking")
