"""Model cards: versions, training-data, validation metrics, refresh date."""

from __future__ import annotations

import os

import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("Model Cards")
st.caption("Live model registry. Cards populate fully in Phase 1 once real PD models exist.")

try:
    v = requests.get(f"{API_URL}/version", timeout=5).json()
    st.json(v)
    st.subheader("Status")
    st.dataframe([
        {"model": "scorecard_pd",  "phase": "Phase 0 (stub)",
         "auc": "tbd Phase 1", "ks": "tbd Phase 1", "psi": "tbd Phase 1"},
        {"model": "xgboost_pd",    "phase": "Phase 0 (stub)",
         "auc": "tbd Phase 1", "ks": "tbd Phase 1", "psi": "tbd Phase 1"},
        {"model": "lgd",           "phase": "Phase 0 (stub)",
         "auc": "n/a (regression)", "ks": "n/a", "psi": "tbd Phase 2"},
        {"model": "vasicek",       "phase": "Phase 0 (stub)",
         "auc": "n/a", "ks": "n/a", "psi": "n/a"},
        {"model": "copula",        "phase": "Phase 0 (stub)",
         "auc": "n/a", "ks": "n/a", "psi": "n/a"},
        {"model": "market_var",    "phase": "Phase 0 (stub)",
         "auc": "n/a", "ks": "n/a", "psi": "n/a"},
    ])
except Exception as exc:
    st.error(f"Cannot reach API: {exc}")
