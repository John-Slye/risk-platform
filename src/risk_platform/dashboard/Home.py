"""Streamlit dashboard entry point."""

from __future__ import annotations

import os
from datetime import date

import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Risk Platform",
    page_icon=":chart_with_downwards_trend:",
    layout="wide",
)

st.title("Integrated Credit & Market Risk Analytics Platform")
st.caption(f"Date: {date.today().isoformat()}  ·  API: `{API_URL}`")

st.markdown(
    """
A production-style risk analytics platform combining market risk and credit
risk in one system. Use the sidebar to navigate.

### What each page does

**1. Market Risk** — Value at Risk and Expected Shortfall across five
methodologies (Historical, Parametric Normal & Student-t, Monte Carlo
MV-Normal & MV-t, Filtered Historical Simulation with GARCH) plus Extreme
Value Theory for the deep tail.
*Default data:* an equal-weighted 9-asset portfolio (SPY, QQQ, IWM, EFA,
TLT, HYG, GLD, USO, UUP) with 2014-2026 daily prices from Yahoo Finance.
*Custom portfolio:* upload a `ticker,weight` CSV (coming soon).

**2. Credit Risk** — Loan-level Probability of Default, Loss Given Default,
Expected Loss, and Basel IRB Risk-Weighted Assets.
*Models:* a logistic-regression scorecard (with WOE/IV feature engineering)
and an XGBoost benchmark, both trained on LendingClub 2014-2016 vintages
(~892K loans).
*Input:* a single loan's features (FICO, DTI, loan amount, term, etc.)
typed into the form.

**3. Portfolio Credit** — Portfolio-level Credit VaR and Economic Capital
via the Vasicek single-factor (ASRF) model and Gaussian / Student-t copula
Monte Carlo (100k+ simulations).
*Input:* parametric (set PD, LGD, asset correlation, number of obligors,
EAD). For heterogeneous portfolios, use the Portfolio Upload page.

**4. Stress Testing** — Re-price the portfolio under historical crisis
windows: 2008 GFC, 2020 COVID, 2022 rate shock.
*Default data:* same 9-asset portfolio. Extended price history downloaded
once for the 2008 scenario.

**5. Model Cards** — Live registry of every trained model: type, training
data, validation AUC/KS/Gini/PSI/Brier, version, refresh date, and
live/fallback status.

**6. Portfolio Upload** — Drop a CSV of loans, get back aggregate Expected
Loss, total Basel RWA, weighted PD/LGD, the top-10 riskiest loans, and a
Plotly scatter of the full portfolio. Download the enriched per-loan
results.
"""
)

# Liveness check
with st.expander("Backend status"):
    try:
        r = requests.get(f"{API_URL}/healthz", timeout=2)
        r.raise_for_status()
        st.success(f"API reachable. Response: `{r.json()}`")
        v = requests.get(f"{API_URL}/version", timeout=2).json()
        st.json(v)
    except Exception as exc:
        st.error(f"Cannot reach API at {API_URL}. Is `docker compose up` running?\n\n{exc}")
