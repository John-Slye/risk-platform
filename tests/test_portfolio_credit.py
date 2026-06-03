"""Phase 7 math tests: Vasicek ASRF + Gaussian/t copula consistency."""

from __future__ import annotations

import pytest

from risk_platform.portfolio import (
    asrf_loss_distribution, conditional_pd, simulate_portfolio_loss,
)


# ---------- Vasicek conditional PD ------------------------------------------
def test_conditional_pd_monotone_in_alpha():
    """As the stress alpha rises, conditional PD rises (more adverse scenario)."""
    a = conditional_pd(0.05, 0.15, 0.50)
    b = conditional_pd(0.05, 0.15, 0.99)
    c = conditional_pd(0.05, 0.15, 0.999)
    assert a < b < c


def test_conditional_pd_increasing_in_rho():
    """Higher correlation -> fatter joint tail -> higher cond. PD at the 99.9% draw."""
    low_rho  = conditional_pd(0.05, 0.05, 0.999)
    high_rho = conditional_pd(0.05, 0.30, 0.999)
    assert high_rho > low_rho


def test_conditional_pd_in_unit_interval():
    """PD must stay in [0, 1] for any reasonable input."""
    for pd_rate in (0.001, 0.05, 0.20, 0.5):
        for rho in (0.05, 0.15, 0.30):
            for alpha in (0.95, 0.99, 0.999):
                cpd = conditional_pd(pd_rate, rho, alpha)
                assert 0 <= cpd <= 1


# ---------- ASRF analytical -------------------------------------------------
def test_asrf_el_equals_unconditional_el():
    pd, rho, lgd, ead, n = 0.05, 0.15, 0.45, 1_000.0, 100
    out = asrf_loss_distribution(pd=pd, rho=rho, lgd=lgd, ead=ead, n_obligors=n)
    assert out["expected_loss"] == pytest.approx(pd * lgd * ead * n)


def test_asrf_var_greater_than_el():
    out = asrf_loss_distribution(pd=0.05, rho=0.15, lgd=0.45,
                                 ead=1_000, n_obligors=500)
    assert out["VaR_99.9"] > out["expected_loss"]
    assert out["VaR_99.0"] < out["VaR_99.9"]
    assert out["economic_capital_99_9"] > 0


# ---------- Gaussian / Student-t copula MC ----------------------------------
def test_t_copula_fatter_tail_than_gaussian():
    """Student-t copula must produce a heavier joint tail than Gaussian under
    identical marginal PDs. This is the central post-2008 modeling story."""
    args = dict(pds=0.20, eads=10_000, lgds=0.92, rho=0.10,
                n_obligors=500, n_sims=20_000, seed=42)
    g = simulate_portfolio_loss(**args, copula="gaussian")
    t = simulate_portfolio_loss(**args, copula="t", df=5)
    assert t["credit_var_99_9"] > g["credit_var_99_9"]
    # The t-EC premium is typically 15-50% on this kind of book.
    ratio = t["credit_var_99_9"] / g["credit_var_99_9"]
    assert 1.05 < ratio < 2.0


def test_el_agrees_across_copulas():
    """Expected loss is a function of marginals only; copula choice
    cannot change EL. This is a hard invariant."""
    args = dict(pds=0.10, eads=1_000, lgds=0.50, rho=0.15,
                n_obligors=200, n_sims=10_000, seed=0)
    g = simulate_portfolio_loss(**args, copula="gaussian")
    t = simulate_portfolio_loss(**args, copula="t", df=8)
    expected = 0.10 * 0.50 * 1_000 * 200
    assert g["expected_loss"] == pytest.approx(expected)
    assert t["expected_loss"] == pytest.approx(expected)
    assert g["expected_loss"] == pytest.approx(t["expected_loss"])


def test_gaussian_mc_converges_to_vasicek_asymptotic():
    """In the large-portfolio limit, Gaussian copula MC must converge to the
    Vasicek ASRF analytical answer. This is THE consistency check Basel IRB
    is built on."""
    asrf = asrf_loss_distribution(pd=0.20, rho=0.10, lgd=0.92,
                                   ead=15_000, n_obligors=1_000)
    g = simulate_portfolio_loss(pds=0.20, eads=15_000, lgds=0.92, rho=0.10,
                                n_obligors=1_000, n_sims=50_000,
                                copula="gaussian", seed=42)
    # Should agree within ~3% (MC sampling noise at 50k sims)
    rel_err = abs(g["credit_var_99_9"] - asrf["VaR_99.9"]) / asrf["VaR_99.9"]
    assert rel_err < 0.05, f"MC vs ASRF disagree by {rel_err*100:.1f}%"


def test_copula_rejects_bad_choice():
    with pytest.raises(ValueError):
        simulate_portfolio_loss(copula="bogus")
