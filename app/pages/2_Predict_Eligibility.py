import os
import sys

import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (  # noqa: E402
    EMI_SCENARIOS,
    create_application,
    load_classification_artifacts,
    predict_eligibility,
    predict_max_emi,
)

st.set_page_config(page_title="Predict Eligibility | EMIPredict AI", page_icon="✅", layout="wide")
st.title("✅ EMI Eligibility Prediction")
st.caption("Real-time classification: Eligible / High Risk / Not Eligible")

model, encoder, metadata = load_classification_artifacts()
if model is None:
    st.warning("No trained classification model found. Run `python src/train_classification.py` first.")
    st.stop()

st.info(f"Serving predictions from the best-performing model: **{metadata['best_model_name']}** "
        f"(validation accuracy: {metadata['validation_metrics'][metadata['best_model_name']]['accuracy']:.2%})")

with st.form("eligibility_form"):
    st.markdown("#### Applicant Profile")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", 18, 75, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
    with c2:
        monthly_salary = st.number_input("Monthly Salary (INR)", 15_000, 500_000, 45_000, step=1000)
        employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
        years_of_employment = st.number_input("Years of Employment", 0.0, 45.0, 5.0, step=0.5)
        company_type = st.selectbox("Company Type", ["Startup", "SME", "Large Corporate", "MNC", "Government", "Not Applicable"])
    with c3:
        house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
        monthly_rent = st.number_input("Monthly Rent (INR)", 0, 100_000, 10_000, step=500)
        family_size = st.number_input("Family Size", 1, 15, 4)
        dependents = st.number_input("Dependents", 0, 10, 1)
    with c4:
        existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
        current_emi_amount = st.number_input("Current EMI Amount (INR)", 0, 100_000, 0, step=500)
        credit_score = st.slider("Credit Score", 300, 850, 680)
        bank_balance = st.number_input("Bank Balance (INR)", 0, 5_000_000, 80_000, step=1000)

    st.markdown("#### Monthly Financial Obligations")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1:
        school_fees = st.number_input("School Fees", 0, 50_000, 0, step=500)
    with e2:
        college_fees = st.number_input("College Fees", 0, 100_000, 0, step=500)
    with e3:
        travel_expenses = st.number_input("Travel Expenses", 0, 50_000, 2_000, step=500)
    with e4:
        groceries_utilities = st.number_input("Groceries & Utilities", 0, 100_000, 8_000, step=500)
    with e5:
        other_monthly_expenses = st.number_input("Other Expenses", 0, 50_000, 2_000, step=500)

    emergency_fund = st.number_input("Emergency Fund (INR)", 0, 2_000_000, 30_000, step=1000)

    st.markdown("#### Loan Application Details")
    l1, l2, l3 = st.columns(3)
    with l1:
        emi_scenario = st.selectbox("EMI Scenario", EMI_SCENARIOS)
    with l2:
        requested_amount = st.number_input("Requested Amount (INR)", 5_000, 2_000_000, 100_000, step=5000)
    with l3:
        requested_tenure = st.number_input("Requested Tenure (months)", 1, 96, 24)

    customer_name = st.text_input("Customer Name (optional, for saving the record)", "")
    submitted = st.form_submit_button("Predict Eligibility", type="primary", use_container_width=True)

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

    result = predict_eligibility(raw_input)
    emi_result = predict_max_emi(raw_input)

    # Persist across reruns (e.g. the "Save" button click below) so the
    # result stays visible instead of disappearing on the next script run.
    st.session_state["eligibility_prediction"] = {
        "raw_input": raw_input,
        "customer_name": customer_name,
        "result": result,
        "emi_result": emi_result,
    }

if "eligibility_prediction" in st.session_state:
    pred = st.session_state["eligibility_prediction"]
    raw_input = pred["raw_input"]
    customer_name = pred["customer_name"]
    result = pred["result"]
    emi_result = pred["emi_result"]
    requested_amount = raw_input["requested_amount"]
    requested_tenure = raw_input["requested_tenure"]

    st.divider()
    label = result["label"]
    color = {"Eligible": "green", "High_Risk": "orange", "Not_Eligible": "red"}[label]
    st.markdown(f"### Result: :{color}[{label.replace('_', ' ')}]")
    st.progress(result["confidence"], text=f"Model confidence: {result['confidence']:.1%}")

    col1, col2 = st.columns([1, 1])
    with col1:
        probs = result["class_probabilities"]
        fig = px.bar(
            x=list(probs.keys()), y=list(probs.values()),
            color=list(probs.keys()),
            color_discrete_map={"Eligible": "#2E7D32", "High_Risk": "#F9A825", "Not_Eligible": "#C62828"},
            labels={"x": "", "y": "Probability"}, title="Class Probabilities",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Predicted Max Safe Monthly EMI", f"₹{emi_result['predicted_max_emi']:,.0f}")
        req_emi_ratio = requested_amount / max(requested_tenure, 1)
        st.write(f"**Requested amount:** ₹{requested_amount:,.0f} over {requested_tenure} months")
        st.write(f"**Model used (classification):** {result['model_name']}")
        st.write(f"**Model used (regression):** {emi_result['model_name']}")

    if st.button("💾 Save this application to Data Management"):
        record = {**raw_input, "customer_name": customer_name or "Unnamed Applicant"}
        record["predicted_eligibility"] = label
        record["eligibility_confidence"] = result["confidence"]
        record["predicted_max_emi"] = emi_result["predicted_max_emi"]
        new_id = create_application(record)
        del st.session_state["eligibility_prediction"]
        st.success(f"Saved as application #{new_id}. View it under Data Management.")
