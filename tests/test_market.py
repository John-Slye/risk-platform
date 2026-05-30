"""Phase 0 smoke tests for market risk module stubs."""

import pytest

from risk_platform.market import MarketRisk


def test_var_99_greater_than_var_95():
    m = MarketRisk()
    v95 = m.var("historical", 0.05)["VaR"]
    v99 = m.var("historical", 0.01)["VaR"]
    assert v99 > v95


def test_es_exceeds_var():
    m = MarketRisk()
    v = m.var("historical", 0.05)["VaR"]
    es = m.es("historical", 0.05)["ES"]
    assert es > v


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        MarketRisk().var(method="bogus")


def test_stress_returns_scenario_keys():
    out = MarketRisk().stress("2020_covid")
    assert {"cum_loss", "worst_day", "ann_vol"}.issubset(out)
