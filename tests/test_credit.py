"""Phase 0 smoke tests for credit module stubs."""

import pytest

from risk_platform.credit import (
    LGDModel, ScorecardPD, XGBoostPD, basel_rwa, expected_loss,
)


def test_scorecard_pd_returns_probability():
    m = ScorecardPD()
    pd = m.predict_proba({"fico": 720})
    assert 0 < pd < 1
    assert isinstance(m.score({}), int)


def test_xgboost_pd_returns_probability():
    pd = XGBoostPD().predict_proba({"fico": 720})
    assert 0 < pd < 1


def test_lgd_returns_in_unit_interval():
    lgd = LGDModel().predict({})
    assert 0 <= lgd <= 1


def test_expected_loss_formula():
    assert expected_loss(0.05, 0.45, 10_000) == pytest.approx(225.0)


def test_basel_rwa_keys_and_signs():
    out = basel_rwa(pd=0.05, lgd=0.45, ead=10_000, maturity_years=3.0)
    assert set(out) == {"K", "RWA", "asset_correlation",
                        "maturity_adjustment", "conditional_pd"}
    assert out["RWA"] > 0
    assert 0 < out["K"] < 1   # capital requirement as fraction of EAD
    assert 0 < out["conditional_pd"] < 1
    assert out["conditional_pd"] > 0.05    # stress should worsen the PD
