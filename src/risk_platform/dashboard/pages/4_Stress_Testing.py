"""Combined stress test: market + credit metrics under historical scenarios."""

from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Stress Testing — combined market + credit impact")
st.caption("Phase 0 wires the market stress endpoint. Phase 4 adds credit overlay.")

scenario = st.selectbox("Scenario", ["2020_covid", "2008_gfc", "2022_rates"])

if st.button("Run scenario"):
    try:
        out = requests.post(f"{API_URL}/market/stress",
                            json={"scenario": scenario}, timeout=5).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Market cum. loss", f"{out['cum_loss']*100:.1f}%")
        c2.metric("Worst day",        f"{out['worst_day']*100:.2f}%")
        c3.metric("Ann. vol",         f"{out['ann_vol']*100:.1f}%")
        st.info(
            "Credit overlay coming in Phase 4 — same scenarios applied to the "
            "loan book via the PD models, with portfolio Credit VaR re-priced."
        )
    except Exception as exc:
        st.error(f"Request failed: {exc}")
