"""Market risk page: pick a method + alpha, get VaR / ES, stress scenarios."""

from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Market Risk — VaR & Expected Shortfall")
st.caption("Phase 0 stub returns Project 1's empirical numbers. Phase 4 wires the real engine.")

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox(
        "Method",
        ["historical", "parametric_normal", "parametric_t",
         "monte_carlo_normal", "monte_carlo_t", "fhs"],
    )
with col2:
    confidence = st.selectbox("Confidence", ["95%", "99%"])
    alpha = 0.05 if confidence == "95%" else 0.01

if st.button("Compute VaR + ES"):
    try:
        body = {"method": method, "alpha": alpha}
        var = requests.post(f"{API_URL}/market/var", json=body, timeout=5).json()
        es  = requests.post(f"{API_URL}/market/es",  json=body, timeout=5).json()
        c1, c2 = st.columns(2)
        c1.metric(f"{confidence} 1-day VaR", f"{var['VaR']*100:.2f}%")
        c2.metric(f"{confidence} 1-day ES",  f"{es['VaR']*100:.2f}%")
        st.caption(f"model: `{var['model']}`")
    except Exception as exc:
        st.error(f"Request failed: {exc}")

st.divider()
st.subheader("Historical stress scenarios")

scenario = st.selectbox("Scenario", ["2020_covid", "2008_gfc", "2022_rates"])
if st.button("Run stress scenario"):
    try:
        out = requests.post(f"{API_URL}/market/stress", json={"scenario": scenario},
                            timeout=5).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative loss", f"{out['cum_loss']*100:.1f}%")
        c2.metric("Worst day",       f"{out['worst_day']*100:.2f}%")
        c3.metric("Ann. vol",        f"{out['ann_vol']*100:.1f}%")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
