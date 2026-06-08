"""Stress testing: re-price the portfolio under historical crisis windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _portfolio_widget import market_portfolio_picker  # noqa: E402


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Stress Testing")

st.markdown(
    """
Re-price the portfolio under actual daily returns observed during real
historical crisis windows. Answers the question *"what would my current
book lose if 2008/2020/2022 happened again?"*

Choose a portfolio in the sidebar: the bundled 9-asset book (default) or
upload your own `ticker,weight` CSV. Extended history (2007+) is downloaded
once for the 2008 GFC scenario.

**Output per scenario:** cumulative loss, worst single day, annualized
volatility within the window.
"""
)

portfolio = market_portfolio_picker()

scenario_labels = {
    "2020_covid": "2020 COVID Crash (Feb 19 - Mar 23)",
    "2008_gfc":   "2008 Global Financial Crisis (Sep - Nov)",
    "2022_rates": "2022 Rate Shock (Jan - Jun)",
}
scenario = st.selectbox("Scenario", list(scenario_labels.keys()),
                        format_func=lambda k: scenario_labels[k])

if st.button("Run scenario"):
    body = {"scenario": scenario}
    if portfolio is not None:
        body["portfolio"] = portfolio
    try:
        out = requests.post(f"{API_URL}/market/stress",
                            json=body, timeout=90).json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Cumulative loss", f"{out['cum_loss']*100:.1f}%")
        c2.metric("Worst day",       f"{out['worst_day']*100:.2f}%")
        c3.metric("Annualized vol",  f"{out['ann_vol']*100:.1f}%")
        st.caption(f"model: `{out['model']}`")
    except Exception as exc:
        st.error(f"Request failed: {exc}")
