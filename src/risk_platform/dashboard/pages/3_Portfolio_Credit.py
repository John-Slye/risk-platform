"""Portfolio credit risk page: copula choice, Credit VaR, Economic Capital."""

from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Portfolio Credit Risk — Credit VaR & Economic Capital")
st.caption("Phase 0 stub. Phase 3 replaces with real Vasicek + Cholesky-MC copula.")

with st.form("port_form"):
    c1, c2, c3 = st.columns(3)
    pd_rate     = c1.number_input("Avg PD",          min_value=0.001, max_value=0.5,  value=0.05, step=0.005)
    rho         = c2.number_input("Asset correlation", min_value=0.01, max_value=0.99, value=0.15, step=0.01)
    n_obligors  = c3.number_input("Number of obligors", min_value=10,  max_value=100_000, value=1000)
    c4, c5, c6 = st.columns(3)
    lgd         = c4.number_input("LGD",              min_value=0.0,   max_value=1.0,  value=0.45, step=0.05)
    copula      = c5.selectbox("Copula", ["gaussian", "t"])
    df          = c6.number_input("t df",             min_value=3,     max_value=30,   value=5, disabled=(copula == "gaussian"))
    n_sims      = st.slider("MC simulations", 1_000, 100_000, 10_000, step=1_000)
    submitted = st.form_submit_button("Simulate portfolio")

if submitted:
    body = dict(pd=pd_rate, rho=rho, n_obligors=n_obligors,
                lgd=lgd, copula=copula, df=df, n_simulations=n_sims)
    try:
        r = requests.post(f"{API_URL}/portfolio/credit_var", json=body, timeout=10)
        r.raise_for_status()
        out = r.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Loss",      f"${out['expected_loss']:,.0f}")
        c2.metric("Credit VaR (99.9%)", f"${out['credit_var_99_9']:,.0f}")
        c3.metric("Economic Capital",   f"${out['economic_capital']:,.0f}")
        st.caption(f"Copula: **{out['copula']}**  ·  df: {out['df']}  ·  sims: {out['n_simulations']:,}")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
