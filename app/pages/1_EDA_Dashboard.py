import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import ROOT_DIR  # noqa: E402

st.set_page_config(page_title="EDA Dashboard | EMIPredict AI", page_icon="\U0001F4CA", layout="wide")
st.title("\U0001F4CA Exploratory Data Analysis Dashboard")
st.caption("Interactive view of the 400,000-record EMI dataset (sampled for responsiveness)")

SAMPLE_PATH = os.path.join(ROOT_DIR, "data", "raw", "emi_dataset_sample.csv")
REPORT_PATH = os.path.join(ROOT_DIR, "reports", "eda_report.md")


@st.cache_data(show_spinner=False)
def load_sample():
    if not os.path.exists(SAMPLE_PATH):
        return None
    return pd.read_csv(SAMPLE_PATH)


df = load_sample()

if df is None:
    st.warning("Sample dataset not found. Run `python src/generate_dataset.py` first.")
    st.stop()

ELIG_ORDER = ["Eligible", "High_Risk", "Not_Eligible"]
ELIG_COLORS = {"Eligible": "#2E7D32", "High_Risk": "#F9A825", "Not_Eligible": "#C62828"}

with st.sidebar:
    st.header("Filters")
    scenario_filter = st.multiselect("EMI Scenario", sorted(df["emi_scenario"].unique()), default=None)
    eligibility_filter = st.multiselect("Eligibility", ELIG_ORDER, default=None)
    salary_range = st.slider(
        "Monthly Salary (INR)", int(df["monthly_salary"].min()), int(df["monthly_salary"].max()),
        (int(df["monthly_salary"].min()), int(df["monthly_salary"].max())),
    )

filtered = df.copy()
if scenario_filter:
    filtered = filtered[filtered["emi_scenario"].isin(scenario_filter)]
if eligibility_filter:
    filtered = filtered[filtered["emi_eligibility"].isin(eligibility_filter)]
filtered = filtered[
    (filtered["monthly_salary"] >= salary_range[0]) & (filtered["monthly_salary"] <= salary_range[1])
]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records (sample)", f"{len(filtered):,}")
c2.metric("Avg Monthly Salary", f"₹{filtered['monthly_salary'].mean():,.0f}")
c3.metric("Avg Max EMI Capacity", f"₹{filtered['max_monthly_emi'].mean():,.0f}")
c4.metric("Eligible Rate", f"{(filtered['emi_eligibility'] == 'Eligible').mean()*100:.1f}%")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Target Distributions", "Financial Patterns", "Demographics", "Full EDA Report"]
)

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        counts = filtered["emi_eligibility"].value_counts().reindex(ELIG_ORDER).fillna(0)
        fig = px.bar(
            x=counts.index, y=counts.values, color=counts.index, color_discrete_map=ELIG_COLORS,
            labels={"x": "", "y": "Count"}, title="Eligibility Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(
            filtered, x="max_monthly_emi", nbins=50, title="Maximum Safe Monthly EMI Distribution",
            color_discrete_sequence=["#1565C0"],
        )
        st.plotly_chart(fig, use_container_width=True)

    ct = pd.crosstab(filtered["emi_scenario"], filtered["emi_eligibility"], normalize="index") * 100
    ct = ct.reindex(columns=ELIG_ORDER).fillna(0).reset_index().melt(
        id_vars="emi_scenario", var_name="emi_eligibility", value_name="pct"
    )
    fig = px.bar(
        ct, x="emi_scenario", y="pct", color="emi_eligibility", color_discrete_map=ELIG_COLORS,
        title="Eligibility Rate by EMI Scenario", labels={"pct": "% of applicants", "emi_scenario": ""},
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(
            filtered.sample(min(3000, len(filtered))), x="monthly_salary", y="max_monthly_emi",
            color="emi_eligibility", color_discrete_map=ELIG_COLORS, opacity=0.5,
            title="Monthly Salary vs Maximum Safe EMI",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(
            filtered, x="emi_eligibility", y="credit_score", color="emi_eligibility",
            category_orders={"emi_eligibility": ELIG_ORDER}, color_discrete_map=ELIG_COLORS,
            title="Credit Score by Eligibility",
        )
        st.plotly_chart(fig, use_container_width=True)

    numeric_cols = [
        "age", "monthly_salary", "credit_score", "bank_balance", "emergency_fund",
        "current_emi_amount", "requested_amount", "max_monthly_emi",
    ]
    corr = filtered[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(filtered, x="age", nbins=30, title="Age Distribution", color_discrete_sequence=["#6A1B9A"])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        ct2 = pd.crosstab(filtered["employment_type"], filtered["emi_eligibility"], normalize="index") * 100
        ct2 = ct2.reindex(columns=ELIG_ORDER).fillna(0).reset_index().melt(
            id_vars="employment_type", var_name="emi_eligibility", value_name="pct"
        )
        fig = px.bar(
            ct2, x="employment_type", y="pct", color="emi_eligibility", color_discrete_map=ELIG_COLORS,
            title="Eligibility Rate by Employment Type", labels={"pct": "% of applicants", "employment_type": ""},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Sample records")
    st.dataframe(filtered.head(200), use_container_width=True, height=300)

with tab4:
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, encoding="utf-8") as f:
            report_md = f.read()
        report_md = report_md.replace("figures/", os.path.join(ROOT_DIR, "reports", "figures") + os.sep)
        st.markdown(report_md)
    else:
        st.info("Run `python src/eda.py` to generate the full EDA report.")
