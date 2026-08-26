# 💳 EMIPredict AI — Intelligent Financial Risk Assessment Platform

A comprehensive **FinTech & Banking** capstone project: an end-to-end machine
learning platform that predicts **EMI eligibility** (classification) and the
**maximum safe monthly EMI amount** (regression) for loan applicants, with
full **MLflow** experiment tracking and a production-style **Streamlit**
web application.

> Solves a real problem: people struggle to repay EMIs due to poor financial
> planning and inadequate risk assessment. This platform gives lenders and
> applicants a data-driven, real-time affordability check before a loan is
> issued.

---

## 1. What this project delivers

| Requirement | Delivered as |
|---|---|
| Dual ML problem solving (classification + regression) | `src/train_classification.py`, `src/train_regression.py` |
| Real-time risk assessment on 400,000 records | `src/generate_dataset.py` (synthetic, realistic, reproducible) |
| Advanced feature engineering (22+ variables) | `src/feature_engineering.py` |
| MLflow experiment tracking + model registry | `sqlite:///mlflow.db` backend, 8 tracked runs + 2 registered models |
| Streamlit Cloud–ready multi-page app | `app/` (6 pages) |
| Complete CRUD for financial data management | `app/pages/5_Data_Management.py` (SQLite-backed) |

---

## 2. Project structure

```
EMI prediction LABMENTIX/
├── app/
│   ├── Home.py                      # Landing page
│   ├── utils.py                     # Model loading, prediction, CRUD (SQLite)
│   └── pages/
│       ├── 1_EDA_Dashboard.py       # Interactive EDA (Plotly)
│       ├── 2_Predict_Eligibility.py # Real-time classification
│       ├── 3_Predict_Max_EMI.py     # Real-time regression
│       ├── 4_Model_Performance.py   # Model comparison + MLflow runs
│       ├── 5_Data_Management.py     # CRUD for loan applications
│       └── 6_About.py               # Project overview
├── src/
│   ├── generate_dataset.py          # Synthesizes the 400K-record dataset
│   ├── preprocessing.py             # Cleaning, quality checks, train/val/test split
│   ├── feature_engineering.py       # Shared feature logic (training + app)
│   ├── eda.py                       # Figures + markdown EDA report
│   ├── model_utils.py               # Shared pipeline/MLflow/plotting helpers
│   ├── train_classification.py      # 4 classifiers + MLflow tracking + selection
│   └── train_regression.py          # 4 regressors + MLflow tracking + selection
├── data/
│   ├── raw/emi_dataset_sample.csv   # 5,000-row sample (committed, for the EDA page)
│   ├── raw/                         # Full 400K dataset (generated, gitignored)
│   └── processed/                   # train/val/test splits (generated, gitignored)
├── models/                          # Trained pipelines + metadata (committed)
├── reports/                         # EDA report, figures, model comparison reports
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```

---

## 3. Dataset

Since no dataset file was supplied with the brief, `src/generate_dataset.py`
**synthesizes** 400,000 realistic financial profiles (80,000 each across 5
EMI scenarios), using vectorized NumPy simulation of income, expenses,
credit history and a financial-capacity model — so relationships between
features and targets are realistic and learnable, not arbitrary.

**5 EMI scenarios:** E-commerce Shopping, Home Appliances, Vehicle, Personal
Loan, Education — each with its own amount range, tenure range, and interest
rate band.

**22+ input variables** across personal demographics, employment & income,
housing & family, monthly financial obligations, credit history, and loan
application details (see `src/feature_engineering.py` for the full list).

**Targets:**
- `emi_eligibility` (classification): `Eligible` / `High_Risk` / `Not_Eligible`
- `max_monthly_emi` (regression): maximum safe monthly EMI in INR

Realistic data-quality issues (missing values, duplicate rows) are injected
intentionally so the preprocessing pipeline has something genuine to clean.

---

## 4. Quickstart (local)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the 400K-record dataset (~6 seconds)
python src/generate_dataset.py

# 3. Clean, validate, engineer features, and split train/val/test
python src/preprocessing.py

# 4. Run EDA (writes reports/eda_report.md + reports/figures/*.png)
python src/eda.py

# 5. Train & MLflow-track classification models (4 models)
python src/train_classification.py

# 6. Train & MLflow-track regression models (4 models)
python src/train_regression.py

# 7. Launch the app
streamlit run app/Home.py
```

Browse MLflow experiments locally:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 5. Modeling approach

**Classification — EMI Eligibility** (Logistic Regression, Random Forest,
XGBoost, Decision Tree): evaluated on accuracy, precision/recall/F1 (macro),
and ROC-AUC (OvR macro). Best model selected by validation accuracy, then
confirmed on a held-out test set never used for model selection.

**Regression — Maximum Monthly EMI** (Linear Regression, Random Forest,
XGBoost, Decision Tree): evaluated on RMSE, MAE, R², and MAPE. Best model
selected by validation RMSE, confirmed on the held-out test set.

Both pipelines use identical `ColumnTransformer` preprocessing
(one-hot encoding for categoricals, standard scaling for numerics) wrapped in
an sklearn `Pipeline`, so the exact same transformation logic used in
training is applied to a single form submission in the Streamlit app —
no train/serve skew.

See `reports/model_comparison_classification.md` and
`reports/model_comparison_regression.md` for full metrics after training,
and the **Model Performance** page in the app for an interactive view.

---

## 6. Deploying to Streamlit Cloud

1. Push this repository to GitHub (the trained `models/*.pkl` files and
   `reports/` are committed so the app works immediately after deploy —
   no training required on the cloud instance).
2. On [share.streamlit.io](https://share.streamlit.io), create a new app
   pointing at this repo with **main file path**: `app/Home.py`.
3. Streamlit Cloud installs `requirements.txt` automatically and deploys.
4. (Optional) To retrain from scratch on a fresh clone, run the 4 pipeline
   scripts in step 4 above before starting the app.

---

## 7. CRUD / Data Management

The **Data Management** page implements full CRUD over a local SQLite store
(`data/app_data.db`, auto-created) of loan applications:
- **Create** — add a new applicant record (predictions run automatically).
- **Read** — search/filter/export all stored applications.
- **Update** — edit a record, optionally re-run predictions on the new data.
- **Delete** — remove a single record or clear the store.

---

## 8. Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · MLflow · Streamlit ·
Plotly · Matplotlib/Seaborn · SQLite

## 9. Domain

FinTech and Banking — automated loan-approval acceleration, risk-based
pricing, and standardized eligibility criteria across 5 lending scenarios.
