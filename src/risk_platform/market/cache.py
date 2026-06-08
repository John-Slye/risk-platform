"""MarketRisk factory with per-portfolio caching.

The cache key is a stable hash of the (ticker, weight) pairs (sorted) plus an
optional start_date. First-time access for a given portfolio fetches prices
and fits GARCH (~30-60 sec). Subsequent calls hit the in-process cache.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd

from .data import DEFAULT_TICKERS, DEFAULT_WEIGHTS
from .market_risk import MarketRisk


_DEFAULT_KEY = "__default__"
_cache: dict[str, MarketRisk] = {}


def _portfolio_key(
    tickers: list[str], weights: list[float], start: Optional[str] = None
) -> str:
    """Stable hash for a (tickers, weights, start_date) tuple."""
    h = hashlib.sha256()
    for t, w in sorted(zip(tickers, weights)):
        h.update(f"{t}:{round(w, 6)}|".encode())
    if start:
        h.update(start.encode())
    return h.hexdigest()[:16]


def get_market_risk(
    tickers: Optional[list[str]] = None,
    weights: Optional[list[float]] = None,
    start: Optional[str] = None,
) -> MarketRisk:
    """Return a cached MarketRisk instance for the given portfolio.

    Pass `tickers=None, weights=None` to get the default 9-asset engine.
    Weights are normalized to sum to 1 before hashing/caching.
    """
    # Default path
    if tickers is None or weights is None or len(tickers) == 0:
        if _DEFAULT_KEY not in _cache:
            _cache[_DEFAULT_KEY] = MarketRisk(
                tickers=list(DEFAULT_TICKERS), weights=DEFAULT_WEIGHTS.copy(),
            )
        return _cache[_DEFAULT_KEY]

    # Custom portfolio path
    if len(tickers) != len(weights):
        raise ValueError("tickers and weights must have same length")
    if not all(isinstance(t, str) and t for t in tickers):
        raise ValueError("tickers must be non-empty strings")

    # Normalize weights
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    w_norm = [w / total for w in weights]

    key = _portfolio_key(tickers, w_norm, start)
    if key not in _cache:
        weights_series = pd.Series(dict(zip(tickers, w_norm)))
        _cache[key] = MarketRisk(tickers=list(tickers), weights=weights_series)
    return _cache[key]


def clear_cache() -> None:
    """Drop all cached engines. Used by tests."""
    _cache.clear()
