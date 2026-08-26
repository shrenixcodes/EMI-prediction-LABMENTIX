import os
import sys

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (  # noqa: E402
    EMI_SCENARIOS,
    create_application,
    load_regression_artifacts,
    predict_eligibility,
    predict_max_emi,
)
from feature_engineering import SCENARIO_RATE_TABLE  # noqa: E402


def emi_formula(principal, annual_rate, tenure_months):
    r = annual_rate / 12.0
    factor = (1 + r) ** tenure_months
    return principal * r * factor / (factor - 1)


st.set_page_config(page_title="Predict Max EMI | EMIPredict AI", page_icon="\U0001F4B0", layout="wide")
st.title("\U0001F4B0 Maximum Safe Monthly EMI Prediction")
st.caption("Real-time regression: how much EMI can this applicant safely afford per month?")

model, metadata = load_regression_artifacts()
if model is None:
    st.warning("No trained regression model found. Run `python src/train_regression.py` first.")
    st.stop()

st.info(f"Serving predictions from the best-performing model: **{metadata['best_model_name']}** "
        f"(validation RMSE: ₹{metadata['validation_metrics'][metadata['best_model_name']]['rmse']:,.0f})")

with st.form("regression_form"):
    st.markdown("#### Applicant Profile")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", 18, 75, 32)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    with c2:
        monthly_salary = st.number_input("Monthly Salary (INR)", 15_000, 500_000, 60_000, step=1000)
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        years_of_employment = st.number_input("Years of Employment", 0.0, 45.0, 6.0, step=0.5)
        company_type = st.selectbox("Company Type", ["Startup", "SME", "Large Corporate", "MNC", "Government", "Not Applicable"])
    with c3:
        house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
        monthly_rent = st.number_input("Monthly Rent (INR)", 0, 100_000, 12_000, step=500)
        family_size = st.number_input("Family Size", 1, 15, 3)
        dependents = st.number_input("Dependents", 0, 10, 1)
    with c4:
        existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
        current_emi_amount = st.number_input("Current EMI Amount (INR)", 0, 100_000, 0, step=500)
        credit_score = st.slider("Credit Score", 300, 850, 700)
        bank_balance = st.number_input("Bank Balance (INR)", 0, 5_000_000, 150_000, step=1000)

    st.markdown("#### Monthly Financial Obligations")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        school_fees = st.number_input("School Fees", 0, 50_000, 0, step=500)
    with e2:
        college_fees = st.number_input("College Fees", 0, 100_000, 0, step=500)
    with e3:
        travel_expenses = st.number_input("Travel Expenses", 0, 50_000, 2_500, step=500)
    with e4:
        groceries_utilities = st.number_input("Groceries & Utilities", 0, 100_000, 9_000, step=500)
    with e5:
        other_monthly_expenses = st.number_input("Other Expenses", 0, 50_000, 2_000, step=500)

    emergency_fund = st.number_input("Emergency Fund (INR)", 0, 2_000_000, 60_000, step=1000)

    st.markdown("#### Loan Application Details")
    l1, l2, l3 = st.columns(3)
    with l1:
        emi_scenario = st.selectbox("EMI Scenario", EMI_SCENARIOS)
    with l2:
        requested_amount = st.number_input("Requested Amount (INR)", 5_000, 2_000_000, 150_000, step=5000)
    with l3:
        requested_tenure = st.number_input("Requested Tenure (months)", 1, 96, 36)

    customer_name = st.text_input("Customer Name (optional, for saving the record)", "")
    submitted = st.form_submit_button("Predict Maximum EMI", type="primary", use_container_width=True)

if submitted:
    raw_input = dict(
        age=age, gender=gender, marital_status=marital_status, education=education,
        monthly_salary=monthly_salary, employment_type=employment_type,
        years_of_employment=years_of_employment, company_type=company_type,
        house_type=house_type, monthly_rent=monthly_rent, family_size=family_size,
        dependents=dependents, school_fees=school_fees, college_fees=college_fees,
        travel_expenses=travel_expenses, groceries_utilities=groceries_utilities,
        other_monthly_expenses=other_monthly_expenses, existing_loans=existing_loans,
        current_emi_amount=current_emi_amount, credit_score=credit_score,
        bank_balance=bank_balance, emergency_fund=emergency_fund, emi_scenario=emi_scenario,
        requested_amount=requested_amount, requested_tenure=requested_tenure,
    )

    reg_result = predict_max_emi(raw_input)
    elig_result = predict_eligibility(raw_input)
    max_emi = reg_result["predicted_max_emi"]

    # Persist across reruns (e.g. the "Save" button click below) so the
    # result stays visible instead of disappearing on the next script run.
    st.session_state["max_emi_prediction"] = {
        "raw_input": raw_input,
        "customer_name": customer_name,
        "reg_result": reg_result,
        "elig_result": elig_result,
        "max_emi": max_emi,
    }

if "max_emi_prediction" in st.session_state:
    pred = st.session_state["max_emi_prediction"]
    raw_input = pred["raw_input"]
    customer_name = pred["customer_name"]
    reg_result = pred["reg_result"]
    elig_result = pred["elig_result"]
    max_emi = pred["max_emi"]
    requested_amount = raw_input["requested_amount"]
    requested_tenure = raw_input["requested_tenure"]
    emi_scenario = raw_input["emi_scenario"]

    annual_rate = SCENARIO_RATE_TABLE.get(emi_scenario, 0.14)
    requested_emi = emi_formula(requested_amount, annual_rate, requested_tenure)

    st.divider()
    col1, col2 = st.columns([1, 1.2])
    with col1:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=max_emi,
                number={"prefix": "₹"},
                title={"text": "Maximum Safe Monthly EMI"},
                gauge={
                    "axis": {"range": [0, max(50_000, max_emi * 1.3, requested_emi * 1.3)]},
                    "bar": {"color": "#1565C0"},
                    "steps": [
                        {"range": [0, requested_emi], "color": "#FFCDD2" if requested_emi > max_emi else "#C8E6C9"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.8,
                        "value": requested_emi,
                    },
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red marker = estimated EMI for the requested amount/tenure at a standard scenario interest rate")
    with col2:
        st.metric("Predicted Max Safe Monthly EMI", f"₹{max_emi:,.0f}")
        st.metric("Estimated EMI for Requested Loan", f"₹{requested_emi:,.0f}")
        gap = max_emi - requested_emi
        if gap >= 0:
            st.success(f"Requested EMI is within capacity by ₹{gap:,.0f}/month.")
        else:
            st.error(f"Requested EMI exceeds safe capacity by ₹{-gap:,.0f}/month.")
        label = elig_result["label"]
        color = {"Eligible": "green", "High_Risk": "orange", "Not_Eligible": "red"}[label]
        st.markdown(f"**Corresponding eligibility verdict:** :{color}[{label.replace('_', ' ')}] "
                    f"({elig_result['confidence']:.1%} confidence)")
        st.write(f"**Model used:** {reg_result['model_name']}")

        max_affordable_amount = None
        r = annual_rate / 12.0
        n = requested_tenure
        if r > 0:
            max_affordable_amount = max_emi * ((1 + r) ** n - 1) / (r * (1 + r) ** n)
        if max_affordable_amount:
            st.info(f"At this tenure, the applicant could safely borrow up to "
                    f"**₹{max_affordable_amount:,.0f}** for this EMI scenario.")

    if st.button("💾 Save this application to Data Management"):
        record = {**raw_input, "customer_name": customer_name or "Unnamed Applicant"}
        record["predicted_eligibility"] = elig_result["label"]
        record["eligibility_confidence"] = elig_result["confidence"]
        record["predicted_max_emi"] = max_emi
        new_id = create_application(record)
        del st.session_state["max_emi_prediction"]
        st.success(f"Saved as application #{new_id}. View it under Data Management.")
