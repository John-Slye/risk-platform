"""Single-factor Vasicek (ASRF) model — stub for Phase 0."""

from __future__ import annotations


def vasicek_loss_distribution(
    pd: float = 0.05, rho: float = 0.15, n_obligors: int = 1000, ead: float = 1.0,
    lgd: float = 0.45,
) -> dict:
    """Returns expected loss, 99.9% VaR, and economic capital. Stub returns
    sensible-looking but hardcoded values that scale roughly with inputs."""
    expected_loss = pd * lgd * ead * n_obligors
    var_99_9 = expected_loss * 3.5  # rough scaling — replaced in Phase 3
    economic_capital = var_99_9 - expected_loss
    return {
        "expected_loss": expected_loss,
        "credit_var_99_9": var_99_9,
        "economic_capital": economic_capital,
        "model": "vasicek-stub-0.1.0",
    }
