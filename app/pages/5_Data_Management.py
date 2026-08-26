import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (  # noqa: E402
    EMI_SCENARIOS,
    create_application,
    delete_all_applications,
    delete_application,
    get_application,
    predict_eligibility,
    predict_max_emi,
    read_applications,
    update_application,
)

st.set_page_config(page_title="Data Management | EMIPredict AI", page_icon="\U0001F5C2", layout="wide")
st.title("\U0001F5C2️ Financial Data Management (CRUD)")
st.caption("Create, read, update, and delete stored loan-application records — the operational data store behind the platform.")

tab_view, tab_create, tab_edit, tab_admin = st.tabs(["\U0001F4CB View / Search", "➕ Create", "✏️ Update / Delete", "⚙️ Admin"])

# --------------------------------------------------------------------------- #
with tab_view:
    search = st.text_input("Search by customer name, EMI scenario, or eligibility", "")
    df = read_applications(search or None)
    st.metric("Total stored applications", len(df))
    if df.empty:
        st.info("No applications stored yet. Add one from the **Create** tab, or save a prediction from the "
                "Predict Eligibility / Predict Max EMI pages.")
    else:
        display_cols = [
            "id", "customer_name", "emi_scenario", "requested_amount", "requested_tenure",
            "credit_score", "predicted_eligibility", "eligibility_confidence", "predicted_max_emi",
            "created_at",
        ]
        st.dataframe(df[display_cols], use_container_width=True, height=420)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download all records as CSV", csv, "loan_applications.csv", "text/csv")

# --------------------------------------------------------------------------- #
with tab_create:
    st.markdown("Add a new loan-application record directly (predictions run automatically on save).")
    with st.form("create_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            customer_name = st.text_input("Customer Name", "New Applicant")
            age = st.number_input("Age", 18, 75, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married"])
            education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
            company_type = st.selectbox("Company Type", ["Startup", "SME", "Large Corporate", "MNC", "Government", "Not Applicable"])
            house_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
        with c2:
            monthly_salary = st.number_input("Monthly Salary", 15_000, 500_000, 40_000, step=1000)
            years_of_employment = st.number_input("Years of Employment", 0.0, 45.0, 4.0, step=0.5)
            monthly_rent = st.number_input("Monthly Rent", 0, 100_000, 8_000, step=500)
            family_size = st.number_input("Family Size", 1, 15, 3)
            dependents = st.number_input("Dependents", 0, 10, 1)
            existing_loans = st.selectbox("Existing Loans", ["No", "Yes"])
            current_emi_amount = st.number_input("Current EMI Amount", 0, 100_000, 0, step=500)
            credit_score = st.slider("Credit Score", 300, 850, 650)
        with c3:
            bank_balance = st.number_input("Bank Balance", 0, 5_000_000, 50_000, step=1000)
            emergency_fund = st.number_input("Emergency Fund", 0, 2_000_000, 20_000, step=1000)
            school_fees = st.number_input("School Fees", 0, 50_000, 0, step=500)
            college_fees = st.number_input("College Fees", 0, 100_000, 0, step=500)
            travel_expenses = st.number_input("Travel Expenses", 0, 50_000, 2_000, step=500)
            groceries_utilities = st.number_input("Groceries & Utilities", 0, 100_000, 7_000, step=500)
            other_monthly_expenses = st.number_input("Other Expenses", 0, 50_000, 1_500, step=500)

        l1, l2, l3 = st.columns(3)
        with l1:
            emi_scenario = st.selectbox("EMI Scenario", EMI_SCENARIOS)
        with l2:
            requested_amount = st.number_input("Requested Amount", 5_000, 2_000_000, 80_000, step=5000)
        with l3:
            requested_tenure = st.number_input("Requested Tenure (months)", 1, 96, 18)

        create_submitted = st.form_submit_button("Create Record", type="primary", use_container_width=True)

    if create_submitted:
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
        elig = predict_eligibility(raw_input)
        reg = predict_max_emi(raw_input)
        record = {
            **raw_input,
            "customer_name": customer_name,
            "predicted_eligibility": elig["label"] if elig else None,
            "eligibility_confidence": elig["confidence"] if elig else None,
            "predicted_max_emi": reg["predicted_max_emi"] if reg else None,
        }
        new_id = create_application(record)
        st.success(f"Created application #{new_id} for **{customer_name}** — "
                   f"predicted: {elig['label'] if elig else 'N/A'}")

# --------------------------------------------------------------------------- #
with tab_edit:
    df_all = read_applications()
    if df_all.empty:
        st.info("No records available to update or delete yet.")
    else:
        record_id = st.selectbox("Select application ID", df_all["id"].tolist())
        record = get_application(int(record_id))
        if record:
            st.markdown(f"#### Editing Application #{record['id']} — {record['customer_name']}")
            with st.form("edit_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    customer_name = st.text_input("Customer Name", record["customer_name"])
                    monthly_salary = st.number_input("Monthly Salary", 15_000, 500_000, int(record["monthly_salary"]), step=1000)
                    credit_score = st.slider("Credit Score", 300, 850, int(record["credit_score"]))
                with c2:
                    requested_amount = st.number_input("Requested Amount", 5_000, 2_000_000, int(record["requested_amount"]), step=5000)
                    requested_tenure = st.number_input("Requested Tenure (months)", 1, 96, int(record["requested_tenure"]))
                with c3:
                    current_emi_amount = st.number_input("Current EMI Amount", 0, 100_000, int(record["current_emi_amount"]), step=500)
                    bank_balance = st.number_input("Bank Balance", 0, 5_000_000, int(record["bank_balance"]), step=1000)

                col_a, col_b = st.columns(2)
                update_btn = col_a.form_submit_button("Update Record", type="primary", use_container_width=True)
                recompute_btn = col_b.form_submit_button("Update & Re-run Prediction", use_container_width=True)

            if update_btn or recompute_btn:
                updates = {
                    "customer_name": customer_name,
                    "monthly_salary": monthly_salary,
                    "credit_score": credit_score,
                    "requested_amount": requested_amount,
                    "requested_tenure": requested_tenure,
                    "current_emi_amount": current_emi_amount,
                    "bank_balance": bank_balance,
                }
                if recompute_btn:
                    merged = {**record, **updates}
                    raw_input = {k: merged[k] for k in merged if k in [
                        "age", "gender", "marital_status", "education", "monthly_salary", "employment_type",
                        "years_of_employment", "company_type", "house_type", "monthly_rent", "family_size",
                        "dependents", "school_fees", "college_fees", "travel_expenses", "groceries_utilities",
                        "other_monthly_expenses", "existing_loans", "current_emi_amount", "credit_score",
                        "bank_balance", "emergency_fund", "emi_scenario", "requested_amount", "requested_tenure",
                    ]}
                    elig = predict_eligibility(raw_input)
                    reg = predict_max_emi(raw_input)
                    updates["predicted_eligibility"] = elig["label"] if elig else record["predicted_eligibility"]
                    updates["eligibility_confidence"] = elig["confidence"] if elig else record["eligibility_confidence"]
                    updates["predicted_max_emi"] = reg["predicted_max_emi"] if reg else record["predicted_max_emi"]

                update_application(int(record_id), updates)
                st.success(f"Application #{record_id} updated.")
                st.rerun()

            st.divider()
            if st.button(f"\U0001F5D1️ Delete Application #{record_id}", type="secondary"):
                delete_application(int(record_id))
                st.success(f"Application #{record_id} deleted.")
                st.rerun()

# --------------------------------------------------------------------------- #
with tab_admin:
    st.markdown("#### Danger zone")
    df_all = read_applications()
    st.write(f"Currently storing **{len(df_all)}** application record(s).")
    confirm = st.checkbox("I understand this will permanently delete all stored applications.")
    if st.button("Delete ALL records", disabled=not confirm):
        delete_all_applications()
        st.success("All records deleted.")
        st.rerun()
