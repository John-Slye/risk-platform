"""Loan-level credit risk page: PD, LGD, Expected Loss, Basel RWA."""

from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Credit Risk — Loan-Level PD / LGD / EL")
st.caption("Phase 0 stub returns hardcoded model outputs. Phase 1 replaces with real PD models.")

with st.form("loan_form"):
    c1, c2, c3 = st.columns(3)
    annual_income  = c1.number_input("Annual income ($)",  min_value=1_000, value=80_000)
    loan_amount    = c2.number_input("Loan amount ($)",    min_value=500,   value=15_000)
    interest_rate  = c3.number_input("Interest rate",      min_value=0.01,  max_value=0.99, value=0.105, step=0.005)
    c4, c5, c6 = st.columns(3)
    term_months    = c4.selectbox("Term (months)", [36, 60], index=0)
    fico           = c5.slider("FICO", 300, 850, 700)
    dti            = c6.number_input("DTI",        min_value=0.0,  max_value=0.99, value=0.18, step=0.01)
    c7, c8, c9 = st.columns(3)
    purpose        = c7.selectbox("Purpose", ["debt_consolidation", "credit_card", "home_improvement", "other"])
    home_ownership = c8.selectbox("Home", ["RENT", "MORTGAGE", "OWN"])
    pd_model       = c9.selectbox("PD Model", ["scorecard", "xgboost"])
    submitted = st.form_submit_button("Compute Expected Loss")

if submitted:
    loan = dict(annual_income=annual_income, loan_amount=loan_amount,
                interest_rate=interest_rate, term_months=term_months,
                fico=fico, dti=dti, purpose=purpose, home_ownership=home_ownership)
    try:
        r = requests.post(
            f"{API_URL}/credit/expected_loss?pd_model={pd_model}",
            json=loan, timeout=5,
        )
        r.raise_for_status()
        out = r.json()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PD",  f"{out['pd']*100:.2f}%")
        c2.metric("LGD", f"{out['lgd']*100:.1f}%")
        c3.metric("Expected Loss ($)", f"{out['expected_loss']:,.0f}")
        c4.metric("Basel RWA ($)",     f"{out['rwa']:,.0f}")
        st.caption(f"K (capital / EAD): {out['K']*100:.2f}%")
        st.json(out["model_versions"])
    except Exception as exc:
        st.error(f"Request failed: {exc}")
