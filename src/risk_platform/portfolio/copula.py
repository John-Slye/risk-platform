"""Joint-default simulation via Gaussian / Student-t copula — stub for Phase 0."""

from __future__ import annotations


def simulate_portfolio_losses(
    pd: float = 0.05, rho: float = 0.15, n_obligors: int = 1000,
    n_simulations: int = 10_000, copula: str = "gaussian", df: int = 5,
) -> dict:
    """Returns summary stats of the simulated portfolio loss distribution.
    Stub returns hardcoded numbers. Phase 3 replaces with real Cholesky + MC."""
    if copula not in {"gaussian", "t"}:
        raise ValueError("copula must be 'gaussian' or 't'")
    base = pd * 0.45 * n_obligors
    # t-copula gives fatter tails — we encode that in the stub for realism.
    tail_multiplier = 3.5 if copula == "gaussian" else 4.8
    return {
        "copula": copula,
        "df": df if copula == "t" else None,
        "expected_loss": base,
        "credit_var_99": base * 2.6,
        "credit_var_99_9": base * tail_multiplier,
        "economic_capital": base * (tail_multiplier - 1),
        "n_simulations": n_simulations,
        "model": "copula-stub-0.1.0",
    }
