"""Historical stress scenarios.

Given a portfolio's current weights, re-price it under the actual returns
observed during well-known historical crises. The output is *what the current
portfolio would have lost*, not what the original portfolio at the time did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .data import DATA_DIR, compute_returns, download_prices


# Standard scenario windows. Note: 1987 Black Monday is excluded because none
# of the modern ETFs we use existed in 1987 (SPY launched 1993).
STRESS_SCENARIOS: dict[str, tuple[str, str]] = {
    "2008 Global Financial Crisis (Sep–Nov)": ("2008-09-01", "2008-11-30"),
    "2020 COVID Crash (Feb 19 – Mar 23)":     ("2020-02-19", "2020-03-23"),
    "2022 Rate Shock (Jan–Jun)":              ("2022-01-01", "2022-06-30"),
    "2015 China devaluation (Aug)":           ("2015-08-17", "2015-08-28"),
    "2018 Q4 volatility (Oct–Dec)":           ("2018-10-01", "2018-12-31"),
}


def stress_test(
    weights: pd.Series,
    scenarios: Mapping[str, tuple[str, str]] = STRESS_SCENARIOS,
    extended_start: str = "2007-01-01",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Re-price `weights` under each scenario's actual returns.

    For each scenario, returns:
      - n_days        : # trading days in the window
      - cum_loss_%    : cumulative compounded loss (positive number)
      - worst_day_%   : the single worst day in the window
      - worst_date    : when that worst day occurred
      - ann_vol_%     : annualized vol of the portfolio during the window

    Notes
    -----
    Some ETFs (HYG started Apr 2007, UUP Mar 2007) won't have data for the
    earliest part of `extended_start`; we forward-fill / drop NaNs in the
    window before applying weights, so scenarios are evaluated only on the
    assets that existed at the time.
    """
    cache_path = cache_path or (DATA_DIR / "prices_extended.parquet")
    prices = download_prices(
        weights.index.tolist(),
        start=extended_start,
        cache_path=cache_path,
    )
    returns = compute_returns(prices, kind="simple")

    rows = []
    for name, (start, end) in scenarios.items():
        window = returns.loc[start:end].dropna(how="all")
        if window.empty:
            rows.append({
                "scenario": name, "start": start, "end": end,
                "n_days": 0, "cum_loss_%": np.nan,
                "worst_day_%": np.nan, "worst_date": pd.NaT,
                "ann_vol_%": np.nan,
            })
            continue

        # Use only assets with non-NaN returns on each day
        w = weights.reindex(window.columns).fillna(0.0)
        port_rets = (window * w).sum(axis=1, skipna=True)
        cum_ret = (1 + port_rets).prod() - 1
        rows.append({
            "scenario": name,
            "start": start, "end": end,
            "n_days": len(port_rets),
            "cum_loss_%": -cum_ret * 100,
            "worst_day_%": -port_rets.min() * 100,
            "worst_date": port_rets.idxmin().date(),
            "ann_vol_%": port_rets.std() * np.sqrt(252) * 100,
        })
    return pd.DataFrame(rows)


def asset_contributions_in_scenario(
    weights: pd.Series,
    start: str,
    end: str,
    extended_start: str = "2007-01-01",
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Break down per-asset cumulative loss contributions inside one scenario.

    Useful for the README chart: "which assets drove the 2020 COVID loss?"
    """
    cache_path = cache_path or (DATA_DIR / "prices_extended.parquet")
    prices = download_prices(
        weights.index.tolist(), start=extended_start, cache_path=cache_path
    )
    returns = compute_returns(prices, kind="simple")
    window = returns.loc[start:end]
    w = weights.reindex(window.columns).fillna(0.0)
    # Asset-level cumulative return * weight (small-return approximation:
    # the sum across assets approximately equals portfolio cumulative return).
    asset_cum = (1 + window).prod() - 1
    contrib = asset_cum * w  # asset's contribution to portfolio cumulative ret
    return pd.DataFrame({
        "weight": w,
        "asset_cum_return_%": asset_cum * 100,
        "contribution_to_port_%": contrib * 100,
    }).sort_values("contribution_to_port_%")
