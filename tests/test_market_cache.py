"""Tests for the MarketRisk factory + per-portfolio cache."""

from __future__ import annotations

from risk_platform.market import MarketRisk, clear_cache, get_market_risk


def test_default_returns_same_instance():
    clear_cache()
    a = get_market_risk()
    b = get_market_risk()
    assert a is b, "Default singleton should be cached"


def test_same_portfolio_returns_same_instance():
    clear_cache()
    tickers = ["SPY", "TLT"]
    weights = [0.6, 0.4]
    a = get_market_risk(tickers, weights)
    b = get_market_risk(tickers, weights)
    assert a is b


def test_different_portfolio_returns_different_instance():
    clear_cache()
    a = get_market_risk(["SPY", "TLT"], [0.6, 0.4])
    b = get_market_risk(["SPY", "QQQ"], [0.5, 0.5])
    assert a is not b


def test_normalized_weights_hit_same_cache():
    """[60, 40] and [0.6, 0.4] should both normalize to the same key."""
    clear_cache()
    a = get_market_risk(["SPY", "TLT"], [60.0, 40.0])
    b = get_market_risk(["SPY", "TLT"], [0.6, 0.4])
    assert a is b


def test_ticker_order_doesnt_affect_cache():
    """Sorting in the hash means (A,B) and (B,A) hit the same cache entry."""
    clear_cache()
    a = get_market_risk(["SPY", "TLT"], [0.6, 0.4])
    b = get_market_risk(["TLT", "SPY"], [0.4, 0.6])
    assert a is b


def test_returns_market_risk_instance():
    clear_cache()
    a = get_market_risk(["SPY", "TLT"], [0.6, 0.4])
    assert isinstance(a, MarketRisk)
    assert set(a.tickers) == {"SPY", "TLT"}


def test_mismatched_lengths_raises():
    clear_cache()
    import pytest
    with pytest.raises(ValueError):
        get_market_risk(["SPY", "TLT"], [1.0])


def test_zero_weights_raises():
    clear_cache()
    import pytest
    with pytest.raises(ValueError):
        get_market_risk(["SPY", "TLT"], [0.0, 0.0])
