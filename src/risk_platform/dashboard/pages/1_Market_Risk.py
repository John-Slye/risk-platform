"""Market risk page: VaR / ES across methods, with side-by-side comparison."""

from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Market Risk - VaR & Expected Shortfall")
st.caption("Powered by the Project 1 engine (5 VaR methods + EVT) ported in as `risk_platform.market`.")

METHODS = ["historical", "parametric_normal", "parametric_t",
           "monte_carlo_normal", "monte_carlo_t", "fhs"]

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox("Method", METHODS, index=0)
with col2:
    confidence = st.selectbox("Confidence", ["95%", "99%"])
    alpha = 0.05 if confidence == "95%" else 0.01

if st.button("Compute VaR + ES"):
    try:
        body = {"method": method, "alpha": alpha}
        var = requests.post(f"{API_URL}/market/var", json=body, timeout=10).json()
        es  = requests.post(f"{API_URL}/market/es",  json=body, timeout=10).json()
        c1, c2 = st.columns(2)
        c1.metric(f"{confidence} 1-day VaR", f"{var['VaR']*100:.2f}%")
        c2.metric(f"{confidence} 1-day ES",  f"{es['VaR']*100:.2f}%")
        st.caption(f"model: `{var['model']}`")
    except Exception as exc:
        st.error(f"Request failed: {exc}")

st.divider()
st.subheader("Method comparison")
st.caption(f"Fires {len(METHODS)} API calls and plots them side by side.")

if st.button("Compute all methods at " + confidence):
    rows = []
    progress = st.progress(0.0)
    for i, m in enumerate(METHODS, 1):
        try:
            v = requests.post(f"{API_URL}/market/var",
                              json={"method": m, "alpha": alpha}, timeout=15).json()
            rows.append({"method": m, "VaR": v["VaR"] * 100})
        except Exception as exc:
            st.warning(f"{m}: {exc}")
        progress.progress(i / len(METHODS))
    if rows:
        labels = [r["method"] for r in rows]
        values = [r["VaR"]  for r in rows]
        fig = go.Figure(go.Bar(x=labels, y=values,
                               marker_color="steelblue",
                               text=[f"{v:.2f}%" for v in values],
                               textposition="outside"))
        fig.update_layout(
            title=f"{confidence} 1-day VaR by method",
            yaxis_title="VaR (% of portfolio)",
            height=440,
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Historical stress scenarios")

scenario = st.selectbox("Scenario", ["2020_covid", "2008_gfc", "2022_rates"])
if st.button("Run stress scenario"):
    try:
        out = requests.post(f"{API_URL}/market/stress",
                            json={"scenario": scenario}, timeout=15).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative loss", f"{out['cum_loss']*100:.1f}%")
        c2.metric("Worst day",       f"{out['worst_day']*100:.2f}%")
        c3.metric("Ann. vol",        f"{out['ann_vol']*100:.1f}%")
        st.caption(f"model: `{out['model']}`")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
