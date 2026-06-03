"""Phase 7 math tests: Expected Loss formula + Basel IRB + portfolio aggregator."""

from __future__ import annotations

import pandas as pd
import pytest

from risk_platform.credit import (
    LGDModel, basel_rwa, expected_loss, portfolio_expected_loss,
)
from risk_platform.credit.pd_models import ScorecardPD as StubPD


# ---------- EL formula ------------------------------------------------------
def test_expected_loss_formula():
    assert expected_loss(0.05, 0.45, 10_000) == pytest.approx(225.0)


def test_expected_loss_zero_pd_zero_loss():
    assert expected_loss(0.0, 0.45, 10_000) == 0.0


# ---------- Basel IRB -------------------------------------------------------
def test_basel_rwa_higher_pd_higher_rwa():
    low  = basel_rwa(pd=0.01, lgd=0.45, ead=1_000)
    high = basel_rwa(pd=0.10, lgd=0.45, ead=1_000)
    assert high["RWA"] > low["RWA"]


def test_basel_rwa_keys_present():
    out = basel_rwa(pd=0.05, lgd=0.45, ead=1_000, maturity_years=3)
    assert set(out) == {"K", "RWA", "asset_correlation",
                        "maturity_adjustment", "conditional_pd"}


def test_basel_conditional_pd_exceeds_unconditional():
    """The 99.9% supervisory factor draw always lifts PD."""
    out = basel_rwa(pd=0.05, lgd=0.45, ead=1_000)
    assert out["conditional_pd"] > 0.05


def test_basel_K_in_unit_interval():
    """Capital requirement K = capital / EAD should be in [0, 1]."""
    for pd_rate in (0.01, 0.05, 0.20):
        out = basel_rwa(pd=pd_rate, lgd=0.45, ead=1_000)
        assert 0 < out["K"] < 1


# ---------- Portfolio aggregator --------------------------------------------
def _sample_loans(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "loan_amnt":   [10_000 + 1_000 * i for i in range(n)],
        "term":        ["36 months"] * n,
        "annual_inc":  [60_000] * n,
        "fico":        [700] * n,
        "dti":         [18.0] * n,
        "int_rate":    [10.5] * n,
        "installment": [330.0] * n,
        "delinq_2yrs": [0] * n,
        "inq_last_6mths": [0] * n,
        "open_acc":    [8] * n,
        "pub_rec":     [0] * n,
        "revol_bal":   [10_000] * n,
        "revol_util":  [40.0] * n,
        "total_acc":   [20] * n,
        "mort_acc":    [0] * n,
        "pub_rec_bankruptcies": [0] * n,
        "emp_length":  ["5 years"] * n,
        "home_ownership": ["RENT"] * n,
        "verification_status": ["Verified"] * n,
        "purpose":     ["debt_consolidation"] * n,
        "application_type": ["Individual"] * n,
    })


def test_portfolio_el_with_stub_models_returns_sane_aggregate():
    loans = _sample_loans(5)
    out = portfolio_expected_loss(loans, StubPD(), LGDModel())
    assert out["n_loans"] == 5
    assert out["total_ead"] == loans["loan_amnt"].sum()
    assert out["total_el"] > 0
    assert out["total_rwa"] > out["total_el"]   # capital must exceed mean loss
    assert 0 < out["weighted_pd"] < 1
    assert 0 < out["weighted_lgd"] < 1


def test_portfolio_el_per_loan_lengths():
    loans = _sample_loans(7)
    out = portfolio_expected_loss(loans, StubPD(), LGDModel())
    per = out["per_loan"]
    assert len(per) == 7
    assert set(["pd", "lgd", "ead", "el", "rwa"]).issubset(per.columns)


def test_portfolio_el_handles_nan_in_features():
    """Missing optional features must not crash. The model layer should fill."""
    loans = _sample_loans(3)
    loans.loc[0, "revol_util"] = float("nan")
    loans.loc[1, "mort_acc"] = float("nan")
    out = portfolio_expected_loss(loans, StubPD(), LGDModel())
    assert out["n_loans"] == 3
    assert out["total_el"] > 0   # stub ignores features so result is non-NaN
