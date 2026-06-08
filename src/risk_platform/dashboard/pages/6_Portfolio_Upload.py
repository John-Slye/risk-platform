"""Portfolio Upload: drag-drop a CSV of loans, get per-loan + aggregate EL/RWA.

CSV columns expected (matching the LoanFeatures schema):
  loan_amnt, int_rate, term, annual_inc, fico, dti
  plus the optional ones (defaults apply if missing)
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_URL = os.environ.get("API_URL", "http://localhost:8000")
SAMPLE_CSV = Path(__file__).resolve().parents[4] / "data" / "sample" / "loan_portfolio.csv"

st.title("Portfolio Upload - Aggregate Expected Loss")

st.markdown(
    """
Drop a CSV of loans, choose a PD model, and get back:

- Aggregate metrics: total Expected Loss, total Basel RWA, EAD-weighted PD,
  EAD-weighted LGD
- The 10 riskiest loans by EL
- A Plotly scatter (PD vs LGD, size = EAD, color = EL)
- A downloadable per-loan results CSV (PD, LGD, EAD, EL, RWA per loan)

**CSV columns required:** `loan_amnt`, `int_rate`, `term`, `annual_inc`,
`fico`, `dti`.
**Optional (sensible defaults applied if missing):** `delinq_2yrs`,
`inq_last_6mths`, `open_acc`, `revol_util`, `mort_acc`, `emp_length`,
`home_ownership`, `verification_status`, `purpose`.

Use the sample CSV below if you don't have your own.
"""
)

# ---- Sample CSV download --------------------------------------------------
if SAMPLE_CSV.exists():
    with open(SAMPLE_CSV) as f:
        st.download_button("Download sample 50-loan CSV", f.read(),
                           file_name="loan_portfolio.csv", mime="text/csv")
else:
    st.caption("(Sample CSV not bundled - run `python scripts/generate_sample_portfolio.py` to create it.)")

# ---- Upload + run ---------------------------------------------------------
uploaded = st.file_uploader("Upload loan portfolio CSV", type=["csv"])
pd_model = st.selectbox("PD Model", ["scorecard", "xgboost"], index=0)

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(f"**{len(df):,} loans uploaded.** Preview:")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Run portfolio analysis"):
        # Fill NaN with LoanFeatures defaults so JSON serialization works.
        defaults = {
            "delinq_2yrs": 0, "inq_last_6mths": 0, "open_acc": 8,
            "pub_rec": 0, "revol_bal": 10_000, "revol_util": 40.0,
            "total_acc": 20, "mort_acc": 0, "pub_rec_bankruptcies": 0,
            "emp_length": "5 years", "home_ownership": "RENT",
            "verification_status": "Verified", "purpose": "debt_consolidation",
            "application_type": "Individual", "term": "36 months",
        }
        df_clean = df.copy()
        for col, default in defaults.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].fillna(default)
        # Drop rows missing any REQUIRED field (loan_amnt, int_rate, etc.)
        required = ["loan_amnt", "int_rate", "term", "annual_inc", "fico", "dti"]
        before = len(df_clean)
        df_clean = df_clean.dropna(subset=required)
        if len(df_clean) < before:
            st.warning(f"Dropped {before - len(df_clean)} loans missing required fields.")
        # Build request body: each row -> LoanFeatures dict
        loans = df_clean.to_dict(orient="records")
        with st.spinner(f"Scoring {len(loans):,} loans..."):
            try:
                r = requests.post(
                    f"{API_URL}/credit/portfolio_el",
                    json={"loans": loans, "pd_model": pd_model},
                    timeout=60,
                )
                r.raise_for_status()
                out = r.json()
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.stop()

        st.subheader("Portfolio totals")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Loans", f"{out['n_loans']:,}")
        c2.metric("Total EAD", f"${out['total_ead']:,.0f}")
        c3.metric("Expected Loss", f"${out['total_el']:,.0f}",
                  delta=f"{out['el_pct_of_ead']:.1f}% of EAD", delta_color="off")
        c4.metric("Basel RWA", f"${out['total_rwa']:,.0f}",
                  delta=f"{out['rwa_density']:.0f}% density", delta_color="off")

        c5, c6 = st.columns(2)
        c5.metric("EAD-weighted PD", f"{out['weighted_pd']*100:.2f}%")
        c6.metric("EAD-weighted LGD", f"{out['weighted_lgd']*100:.2f}%")

        # Per-loan DataFrame
        per = pd.DataFrame({
            "ead": out["per_loan_eads"],
            "pd": out["per_loan_pds"],
            "lgd": out["per_loan_lgds"],
            "el": out["per_loan_els"],
            "rwa": out["per_loan_rwas"],
        })

        st.subheader("Top 10 riskiest loans (by EL)")
        st.dataframe(per.nlargest(10, "el").style.format({
            "ead": "${:,.0f}", "pd": "{:.3f}", "lgd": "{:.3f}",
            "el": "${:,.0f}", "rwa": "${:,.0f}",
        }), use_container_width=True)

        st.subheader("PD vs LGD scatter (size = EAD, color = EL)")
        fig = px.scatter(per, x="pd", y="lgd", size="ead", color="el",
                         labels={"pd": "Probability of Default",
                                 "lgd": "Loss Given Default",
                                 "el": "Expected Loss ($)"},
                         color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Expected Loss distribution across loans")
        fig2 = px.histogram(per, x="el", nbins=20,
                            labels={"el": "Expected Loss ($)"})
        st.plotly_chart(fig2, use_container_width=True)

        # Download enriched portfolio
        buf = io.StringIO()
        per.to_csv(buf, index=False)
        st.download_button("Download per-loan results (CSV)", buf.getvalue(),
                           file_name="portfolio_el_results.csv",
                           mime="text/csv")

        st.caption(f"Models used: {out['model_versions']}")
