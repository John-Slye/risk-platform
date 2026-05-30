"""Streamlit dashboard entry point.

Run locally:
    uv run streamlit run src/risk_platform/dashboard/Home.py

In docker-compose the dashboard reaches the API at http://api:8000 (service name).
Locally it falls back to http://localhost:8000.
"""

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
Welcome. This is a production-style risk analytics platform integrating:

- **Market risk** — VaR / ES across Historical, Parametric, Monte Carlo, FHS,
  and EVT methods (Phase 4, ported from the standalone VaR project).
- **Credit risk (loan-level)** — Logistic-regression scorecard with WOE/IV,
  and an XGBoost benchmark. LGD, Expected Loss, Basel IRB RWA.
- **Portfolio credit risk** — Vasicek single-factor model; joint defaults under
  Gaussian and t-copulas; Credit VaR and Economic Capital.
- **Unified risk report** — market + credit metrics in one call.

Use the sidebar to navigate the dashboard pages. Phase 0 ships stub models;
real models land in Phases 1 through 4.
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
