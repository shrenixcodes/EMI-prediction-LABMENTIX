import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import init_db, models_available  # noqa: E402

st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="\U0001F4B3",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

st.title("\U0001F4B3 EMIPredict AI")
st.subheader("Intelligent Financial Risk Assessment Platform")

st.markdown(
    """
EMIPredict AI is an end-to-end machine learning platform that helps financial
institutions and FinTech companies make **data-driven EMI (Equated Monthly
Installment) lending decisions**. It solves two problems simultaneously for
every loan applicant:

- **Classification** — is this applicant *Eligible*, *High Risk*, or *Not
  Eligible* for their requested EMI?
- **Regression** — what is the **maximum monthly EMI** this applicant can
  safely afford, regardless of what they requested?

Built on **400,000 synthetic financial profiles** across 5 real-world EMI
lending scenarios, with full **MLflow experiment tracking**, multi-model
comparison, and a production-style **Streamlit** interface.
"""
)

if not models_available():
    st.warning(
        "Trained models were not found in `models/`. Run the training pipeline first:\n\n"
        "```bash\npython src/generate_dataset.py\npython src/preprocessing.py\n"
        "python src/train_classification.py\npython src/train_regression.py\n```"
    )
else:
    st.success("Models loaded and ready for real-time predictions.")

st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Records Analyzed", "400,000")
col2.metric("Input Features", "22+")
col3.metric("EMI Scenarios", "5")
col4.metric("Models Trained", "8 (4 classification + 4 regression)")

st.divider()

st.markdown("### Explore the platform")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.page_link("pages/1_EDA_Dashboard.py", label="EDA Dashboard", icon="\U0001F4CA")
with c2:
    st.page_link("pages/2_Predict_Eligibility.py", label="Predict Eligibility", icon="✅")
with c3:
    st.page_link("pages/3_Predict_Max_EMI.py", label="Predict Max EMI", icon="\U0001F4B0")
with c4:
    st.page_link("pages/4_Model_Performance.py", label="Model Performance", icon="\U0001F4C8")
with c5:
    st.page_link("pages/5_Data_Management.py", label="Data Management", icon="\U0001F5C2️")

st.divider()

with st.expander("About the EMI scenarios in this dataset"):
    st.markdown(
        """
| Scenario | Amount Range | Tenure Range |
|---|---|---|
| E-commerce Shopping EMI | ₹10K – ₹200K | 3 – 24 months |
| Home Appliances EMI | ₹20K – ₹300K | 6 – 36 months |
| Vehicle EMI | ₹80K – ₹1,500K | 12 – 84 months |
| Personal Loan EMI | ₹50K – ₹1,000K | 12 – 60 months |
| Education EMI | ₹50K – ₹500K | 6 – 48 months |
"""
    )

st.caption("EMIPredict AI · FinTech & Banking Capstone · Built with Python, scikit-learn, XGBoost, MLflow & Streamlit")
