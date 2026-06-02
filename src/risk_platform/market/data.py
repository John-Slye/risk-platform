"""Data loading and returns computation.

The single entry point most callers want is `load_returns()` which:
  1. Downloads daily adjusted close prices from Yahoo Finance (yfinance)
  2. Caches them to `data/prices.parquet` so we don't re-download
  3. Computes daily returns (simple by default, log optionally)

Example
-------
>>> from portfolio_var.data import load_returns, DEFAULT_TICKERS, DEFAULT_WEIGHTS
>>> returns = load_returns()
>>> port = returns @ DEFAULT_WEIGHTS
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Default portfolio — a defensible multi-asset mix used throughout the project.
# Equal-weighted unless you override.
# ---------------------------------------------------------------------------
DEFAULT_TICKERS: list[str] = [
    "SPY",   # US large-cap equity
    "QQQ",   # US tech / Nasdaq
    "IWM",   # US small-cap
    "EFA",   # Developed international equity
    "TLT",   # Long-duration US Treasuries
    "HYG",   # US high-yield credit
    "GLD",   # Gold
    "USO",   # Oil
    "UUP",   # US dollar index
]

# Equal-weight by default. Sum to 1.0.
DEFAULT_WEIGHTS: pd.Series = pd.Series(
    {t: 1.0 / len(DEFAULT_TICKERS) for t in DEFAULT_TICKERS}
)

# Project root = two levels above this file: src/portfolio_var/data.py -> project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
DEFAULT_CACHE: Path = DATA_DIR / "prices.parquet"


def download_prices(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = "2014-01-01",
    end: str | None = None,
    cache_path: Path | None = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download daily adjusted-close prices for `tickers`.

    Parameters
    ----------
    tickers : iterable of str
    start, end : str
        ISO dates (YYYY-MM-DD). `end=None` means "today".
    cache_path : Path or None
        If given and `refresh=False`, load from this parquet if it exists.
    refresh : bool
        Ignore the cache and re-download.

    Returns
    -------
    DataFrame indexed by date, columns are tickers, values are adj close.
    """
    tickers = list(tickers)

    if cache_path and cache_path.exists() and not refresh:
        cached = pd.read_parquet(cache_path)
        if set(tickers).issubset(cached.columns):
            return cached[tickers]

    # auto_adjust=True bakes splits + dividends into the Close column.
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    # yfinance returns a MultiIndex when multiple tickers; flatten to Close-only.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.concat({t: raw[t]["Close"] for t in tickers}, axis=1)
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.dropna(how="all").ffill().dropna()

    if cache_path:
        prices.to_parquet(cache_path)

    return prices


def compute_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Daily returns from a prices DataFrame.

    Parameters
    ----------
    kind : {'simple', 'log'}
        - 'simple': (P_t / P_{t-1}) - 1. Use for portfolio aggregation
          (port_return = weights @ asset_returns holds exactly).
        - 'log': ln(P_t / P_{t-1}). Use for time-series modeling (GARCH).
    """
    if kind == "simple":
        return prices.pct_change().dropna()
    if kind == "log":
        return np.log(prices / prices.shift(1)).dropna()
    raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")


def load_returns(
    tickers: Iterable[str] = DEFAULT_TICKERS,
    start: str = "2014-01-01",
    end: str | None = None,
    kind: str = "simple",
    refresh: bool = False,
) -> pd.DataFrame:
    """Convenience: download (cached) + compute returns in one call."""
    prices = download_prices(tickers, start=start, end=end, refresh=refresh)
    return compute_returns(prices, kind=kind)


def portfolio_returns(
    returns: pd.DataFrame, weights: pd.Series | None = None
) -> pd.Series:
    """Weighted portfolio return series.

    `weights` must be a Series indexed by the same tickers as `returns.columns`.
    If `weights` is None, equal-weight across `returns.columns` is used.
    """
    if weights is None:
        weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    # Reindex to guarantee column alignment.
    w = weights.reindex(returns.columns).fillna(0.0)
    if not np.isclose(w.sum(), 1.0):
        # Don't silently rescale — surface a warning-ish error.
        raise ValueError(f"weights must sum to 1.0, got {w.sum():.4f}")
    return returns @ w


if __name__ == "__main__":
    # Quick sanity check: download data and print head/tail of portfolio returns.
    rets = load_returns(refresh=True)
    port = portfolio_returns(rets, DEFAULT_WEIGHTS)
    print(f"Returns shape: {rets.shape}")
    print(f"Date range:    {rets.index.min().date()} -> {rets.index.max().date()}")
    print(f"Tickers:       {list(rets.columns)}")
    print("\nPortfolio returns (head):")
    print(port.head())
    print("\nPortfolio returns summary:")
    print(port.describe())
