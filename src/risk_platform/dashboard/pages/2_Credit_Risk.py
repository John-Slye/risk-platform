"""Loan-level credit risk page: PD, LGD, Expected Loss, Basel RWA.

Field names match the LendingClub schema used to train the PD models.
"""

from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Credit Risk - Loan-Level PD / LGD / EL")

st.markdown(
    """
Score a single loan through the trained credit models. Get back Probability
of Default, Loss Given Default, Expected Loss, and Basel IRB Risk-Weighted
Assets.

**Models available:**
- *Scorecard (WOE/IV + Logistic Regression)* - regulator-friendly,
  decomposable into per-feature point contributions.
- *XGBoost benchmark* - higher AUC, used as a challenger model.

Both were trained on **LendingClub 2014-2016 vintages (891,754 loans)** with
out-of-time validation on 2018.

**Input:** enter the loan's features below. Required fields are the
highest-IV ones (loan amount, interest rate, term, annual income, FICO, DTI);
the rest carry sensible defaults that you can override in the optional
section.

For aggregating a whole loan book, see the **Portfolio Upload** page.
"""
)

with st.form("loan_form"):
    st.subheader("Required (highest-IV features)")
    c1, c2, c3 = st.columns(3)
    loan_amnt   = c1.number_input("Loan amount ($)",     min_value=500,   value=15_000)
    int_rate    = c2.number_input("Interest rate (%)",   min_value=1.0,   max_value=50.0, value=10.5, step=0.25)
    term        = c3.selectbox("Term", ["36 months", "60 months"], index=0)

    c4, c5, c6 = st.columns(3)
    annual_inc  = c4.number_input("Annual income ($)",   min_value=1_000, value=80_000)
    fico        = c5.slider("FICO", 300, 850, 700)
    dti         = c6.number_input("DTI (%)",             min_value=0.0,   max_value=99.0, value=18.0, step=0.5)

    with st.expander("Optional features (sensible defaults)"):
        c7, c8, c9 = st.columns(3)
        delinq_2yrs        = c7.number_input("Delinquencies 2yrs", value=0, min_value=0)
        inq_last_6mths     = c8.number_input("Inquiries last 6mo", value=0, min_value=0)
        open_acc           = c9.number_input("Open accounts",      value=8, min_value=0)
        c10, c11, c12 = st.columns(3)
        revol_util         = c10.number_input("Revol. utilization (%)", value=40.0, min_value=0.0, max_value=200.0)
        mort_acc           = c11.number_input("Mortgage accounts", value=0, min_value=0)
        emp_length         = c12.selectbox("Employment length", [
            "< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
            "6 years", "7 years", "8 years", "9 years", "10+ years"], index=5)
        c13, c14, c15 = st.columns(3)
        home_ownership     = c13.selectbox("Home", ["RENT", "MORTGAGE", "OWN"], index=0)
        verification_status = c14.selectbox("Verification",
            ["Verified", "Source Verified", "Not Verified"], index=0)
        purpose            = c15.selectbox("Purpose", [
            "debt_consolidation", "credit_card", "home_improvement",
            "major_purchase", "small_business", "other"], index=0)

    pd_model = st.selectbox("PD Model", ["scorecard", "xgboost"], index=0)
    submitted = st.form_submit_button("Compute Expected Loss")

if submitted:
    loan = dict(
        loan_amnt=loan_amnt, int_rate=int_rate, term=term,
        annual_inc=annual_inc, fico=fico, dti=dti,
        delinq_2yrs=delinq_2yrs, inq_last_6mths=inq_last_6mths,
        open_acc=open_acc, revol_util=revol_util, mort_acc=mort_acc,
        emp_length=emp_length, home_ownership=home_ownership,
        verification_status=verification_status, purpose=purpose,
    )
    try:
        r = requests.post(
            f"{API_URL}/credit/expected_loss?pd_model={pd_model}",
            json=loan, timeout=10,
        )
        r.raise_for_status()
        out = r.json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PD",  f"{out['pd']*100:.2f}%")
        c2.metric("LGD", f"{out['lgd']*100:.1f}%")
        c3.metric("Expected Loss ($)", f"{out['expected_loss']:,.0f}")
        c4.metric("Basel RWA ($)",     f"{out['rwa']:,.0f}")
        st.caption(f"Basel K (capital / EAD): {out['K']*100:.2f}%")
        st.json(out["model_versions"])
    except Exception as exc:
        st.error(f"Request failed: {exc}")
