"""Shared sidebar widget for picking the market portfolio.

Used by the Market Risk and Stress Testing pages. Returns either:
  - None (use default 9-asset portfolio)
  - dict {"tickers": [...], "weights": [...]} (custom portfolio)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


SAMPLE_CSV = Path(__file__).resolve().parents[3] / "data" / "sample" / "market_portfolio.csv"


def market_portfolio_picker() -> Optional[dict]:
    """Render the sidebar portfolio picker. Returns a portfolio dict or None."""
    st.sidebar.subheader("Portfolio source")
    mode = st.sidebar.radio(
        "Data",
        ["Default 9-asset book", "Upload custom (ticker, weight)"],
        index=0,
        label_visibility="collapsed",
    )

    if mode.startswith("Default"):
        st.sidebar.caption(
            "Default: equal-weight SPY, QQQ, IWM, EFA, TLT, HYG, GLD, USO, UUP "
            "(2014-2026 daily from Yahoo Finance, cached)."
        )
        return None

    # Custom upload mode
    if SAMPLE_CSV.exists():
        with open(SAMPLE_CSV) as f:
            st.sidebar.download_button(
                "Download sample (6-asset 60/40)",
                f.read(),
                file_name="market_portfolio.csv",
                mime="text/csv",
                use_container_width=True,
            )

    upload = st.sidebar.file_uploader(
        "CSV: `ticker,weight`",
        type=["csv"],
        help="Each row: a Yahoo Finance ticker and a weight. Weights are "
             "normalized to sum to 1.",
    )
    if not upload:
        st.sidebar.info("Awaiting upload. The page will use the default portfolio "
                        "until a CSV is provided.")
        return None

    try:
        df = pd.read_csv(upload)
        if "ticker" not in df.columns or "weight" not in df.columns:
            st.sidebar.error("CSV must have `ticker` and `weight` columns.")
            return None
        df = df.dropna(subset=["ticker", "weight"])
        if len(df) < 2:
            st.sidebar.error("Need at least 2 tickers.")
            return None
        st.sidebar.success(f"{len(df)} tickers loaded.")
        st.sidebar.dataframe(df, hide_index=True, use_container_width=True)
        return {
            "tickers": df["ticker"].astype(str).tolist(),
            "weights": df["weight"].astype(float).tolist(),
        }
    except Exception as exc:
        st.sidebar.error(f"Failed to parse CSV: {exc}")
        return None
