"""Model Cards: production-style registry of every model in the platform.

Reads metrics from a bundled JSON file produced by the training scripts.
Falls back to a placeholder if the file is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")
METRICS_PATH = Path(__file__).resolve().parents[4] / "models" / "metrics.json"

st.title("Model Cards")
st.caption(
    "Per-model documentation of training data, validation metrics, and refresh "
    "date. Pulls from a bundled `models/metrics.json` updated by the training scripts."
)

# ---------- Load metrics ----------------------------------------------------
metrics = {}
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text())
    except Exception as exc:
        st.error(f"Failed to read {METRICS_PATH}: {exc}")
else:
    st.warning(f"No metrics file at {METRICS_PATH}. Run the training scripts to populate.")

# ---------- Live status via the API -----------------------------------------
live_versions = {}
try:
    live_versions = requests.get(f"{API_URL}/version", timeout=2).json().get("models", {})
except Exception:
    pass


def status_for(model_key: str, version_in_card: str) -> str:
    live = live_versions.get(model_key, "")
    if not live:
        return "unknown"
    return "live (trained)" if "stub" not in live.lower() else "fallback (stub)"


# ---------- Summary table ---------------------------------------------------
st.subheader("Registry")

rows = []
key_to_live_field = {
    "scorecard_pd": "scorecard_pd",
    "xgboost_pd": "xgboost_pd",
    "lgd_model": "lgd",
}
for key, card in metrics.items():
    live_field = key_to_live_field.get(key, key)
    status = status_for(live_field, card.get("version", ""))
    rows.append({
        "Model": card.get("display_name", key),
        "Type": card.get("type", ""),
        "Trained on": card.get("trained_on", ""),
        "Version": card.get("version", ""),
        "Last refresh": card.get("last_refresh", ""),
        "Status": status,
    })

if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
else:
    st.info("No models registered.")

# ---------- Per-model detail expanders --------------------------------------
st.subheader("Per-model detail")


def fmt_pct(x):  return f"{x * 100:.2f}%" if isinstance(x, (int, float)) else x
def fmt_num(x):  return f"{x:.4f}" if isinstance(x, (int, float)) else x


# Scorecard
if "scorecard_pd" in metrics:
    c = metrics["scorecard_pd"]
    with st.expander(c["display_name"], expanded=True):
        cols = st.columns(4)
        cols[0].metric("Test AUC",  f"{c['auc_test']:.4f}")
        cols[1].metric("Test KS",   f"{c['ks_test']:.4f}")
        cols[2].metric("Test Gini", f"{c['gini_test']:.4f}")
        cols[3].metric("PSI train->test",
                       f"{c['psi_train_to_test']:.3f}",
                       help=c["psi_interpretation"])
        st.write("**Top features by Information Value:**")
        st.dataframe(pd.DataFrame(c["top_features_by_iv"]),
                     use_container_width=True, hide_index=True)

# XGBoost
if "xgboost_pd" in metrics:
    c = metrics["xgboost_pd"]
    with st.expander(c["display_name"]):
        cols = st.columns(4)
        cols[0].metric("Test AUC",  f"{c['auc_test']:.4f}",
                       delta=f"+{c['vs_scorecard_auc_delta']:.3f} vs scorecard",
                       delta_color="normal")
        cols[1].metric("Test KS",   f"{c['ks_test']:.4f}")
        cols[2].metric("Test Gini", f"{c['gini_test']:.4f}")
        cols[3].metric("Best iter", f"{c['best_iteration']}")
        st.write("**Top features by gain:** " + ", ".join(c["top_features_by_gain"]))

# LGD
if "lgd_model" in metrics:
    c = metrics["lgd_model"]
    with st.expander(c["display_name"]):
        cols = st.columns(3)
        cols[0].metric("Test MAE",  f"{c['mae_test']:.4f}")
        cols[1].metric("Mean pred", f"{c['mean_predicted']:.3f}")
        cols[2].metric("Mean realized (test)", f"{c['mean_realized_test']:.3f}")
        st.caption(c.get("note", ""))

# Market risk
if "market_risk" in metrics:
    c = metrics["market_risk"]
    with st.expander(c["display_name"]):
        st.write(f"**Trained on:** {c['trained_on']}  (n = {c['n_observations']:,})")
        st.write("**Backtest at 99% (out-of-sample):**")
        bt = c["backtest_summary"]
        st.dataframe(pd.DataFrame({
            "Method":              ["FHS GARCH(1,1)", "Historical", "Parametric Normal"],
            "Exceedance rate":     [f"{bt['fhs_garch']['exc_rate_99']*100:.2f}%",
                                    f"{bt['historical']['exc_rate_99']*100:.2f}%",
                                    f"{bt['parametric_normal']['exc_rate_99']*100:.2f}%"],
            "Kupiec p":            [bt["fhs_garch"]["kupiec_p_99"],
                                    bt["historical"]["kupiec_p_99"],
                                    bt["parametric_normal"]["kupiec_p_99"]],
            "Christoffersen-CC p": [bt["fhs_garch"]["christoffersen_cc_p_99"],
                                    bt["historical"]["christoffersen_cc_p_99"],
                                    bt["parametric_normal"]["christoffersen_cc_p_99"]],
        }), use_container_width=True, hide_index=True)
        st.write(f"**EVT shape parameter (ξ):** {c['evt_xi']:.3f}   (positive = heavy tail confirmed)")

# Portfolio credit
if "portfolio_credit" in metrics:
    c = metrics["portfolio_credit"]
    with st.expander(c["display_name"]):
        st.write(f"**Consistency check:** {c['consistency_check']}")
        st.write(f"**Headline finding:** {c['headline_finding']}")
