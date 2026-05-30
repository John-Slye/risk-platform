"""Phase 0 smoke tests for portfolio credit module stubs."""

import pytest

from risk_platform.portfolio import (
    simulate_portfolio_losses, vasicek_loss_distribution,
)


def test_vasicek_keys_and_inequalities():
    out = vasicek_loss_distribution(pd=0.05, n_obligors=1000)
    assert out["credit_var_99_9"] > out["expected_loss"]
    assert out["economic_capital"] == out["credit_var_99_9"] - out["expected_loss"]


def test_copula_t_has_fatter_tail_than_gaussian():
    g = simulate_portfolio_losses(copula="gaussian")
    t = simulate_portfolio_losses(copula="t")
    assert t["credit_var_99_9"] > g["credit_var_99_9"], \
        "Student-t should imply a heavier tail than Gaussian"


def test_copula_rejects_bad_choice():
    with pytest.raises(ValueError):
        simulate_portfolio_losses(copula="unknown")
