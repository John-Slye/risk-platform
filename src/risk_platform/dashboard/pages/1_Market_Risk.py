"""Market risk page: VaR / ES across methods, with side-by-side comparison."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st

# Import shared widget from the dashboard package root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portfolio_widget import market_portfolio_picker  # noqa: E402


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Market Risk - VaR & Expected Shortfall")

st.markdown(
    """
Compute 1-day Value at Risk and Expected Shortfall on a multi-asset
portfolio using five methodologies plus Extreme Value Theory.

Choose a portfolio in the sidebar: the bundled 9-asset book (default) or
upload your own `ticker,weight` CSV. First call on a custom portfolio
takes ~30-60 sec (yfinance download + GARCH fit); subsequent calls on the
same portfolio are instant thanks to the in-process cache.

Pick a method and confidence below to compute VaR + ES. Use **Compute all
methods** to plot all five side-by-side and see the fat-tail gap directly.
"""
)

portfolio = market_portfolio_picker()

METHODS = ["historical", "parametric_normal", "parametric_t",
           "monte_carlo_normal", "monte_carlo_t", "fhs"]

col1, col2 = st.columns(2)
with col1:
    method = st.selectbox("Method", METHODS, index=0)
with col2:
    confidence = st.selectbox("Confidence", ["95%", "99%"])
    alpha = 0.05 if confidence == "95%" else 0.01

def _body(method: str, alpha: float) -> dict:
    body = {"method": method, "alpha": alpha}
    if portfolio is not None:
        body["portfolio"] = portfolio
    return body


if st.button("Compute VaR + ES"):
    try:
        var = requests.post(f"{API_URL}/market/var",
                            json=_body(method, alpha), timeout=90).json()
        es  = requests.post(f"{API_URL}/market/es",
                            json=_body(method, alpha), timeout=90).json()
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
                              json=_body(m, alpha), timeout=120).json()
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
