"""Vasicek single-factor (ASRF) credit model.

The asymptotic-single-risk-factor model is the foundation of the Basel IRB
formula. Each obligor's latent 'asset value' depends on a single systematic
factor plus an idiosyncratic shock. In the large-portfolio limit, the
portfolio loss distribution has a closed form.

A_i = sqrt(rho) * M + sqrt(1-rho) * eps_i,  M, eps_i ~ N(0, 1)
Default if A_i < Phi^-1(PD_i).

Conditional PD given the supervisory-worst-case alpha-quantile factor draw:
    PD_alpha = Phi( ( Phi^-1(PD) + sqrt(rho) * Phi^-1(alpha) ) / sqrt(1 - rho) )

Credit VaR_alpha (asymptotic, homogeneous portfolio) = EAD * LGD * PD_alpha.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def conditional_pd(pd: float, rho: float, alpha: float = 0.999) -> float:
    """Vasicek conditional PD under an alpha-quantile factor draw.

    Higher alpha = more adverse scenario. Basel IRB uses alpha = 0.999.
    """
    return float(
        norm.cdf(
            (norm.ppf(pd) + np.sqrt(rho) * norm.ppf(alpha)) / np.sqrt(1 - rho)
        )
    )


def asrf_loss_distribution(
    pd: float = 0.05,
    rho: float = 0.15,
    lgd: float = 0.45,
    ead: float = 1.0,
    n_obligors: int = 1_000,
    alphas: tuple[float, ...] = (0.99, 0.999),
) -> dict:
    """Analytical Vasicek/ASRF loss summary for a homogeneous portfolio.

    Inputs are scalar (every obligor has identical PD/LGD/EAD).
    For heterogeneous portfolios use `portfolio.copula.simulate_*` instead.

    Returns expected loss, conditional PD per alpha, Credit VaR per alpha,
    and Economic Capital (Credit VaR - EL) per alpha.
    """
    total_ead = ead * n_obligors
    el = pd * lgd * total_ead

    var_per_alpha = {}
    for alpha in alphas:
        cpd = conditional_pd(pd, rho, alpha)
        var_per_alpha[f"VaR_{int(alpha*1000)/10}"] = cpd * lgd * total_ead
        var_per_alpha[f"cond_pd_{int(alpha*1000)/10}"] = cpd

    out = {
        "expected_loss": el,
        "total_ead": total_ead,
        "rho": rho,
        "n_obligors": n_obligors,
        "model": "vasicek-asrf-0.1.0",
    }
    out.update(var_per_alpha)
    out["economic_capital_99_9"] = out["VaR_99.9"] - el
    return out


# Backward-compat shim: old stub callers used this name. Now returns the
# analytical Vasicek result instead of hardcoded numbers.
def vasicek_loss_distribution(
    pd: float = 0.05, rho: float = 0.15, n_obligors: int = 1_000,
    ead: float = 1.0, lgd: float = 0.45,
) -> dict:
    out = asrf_loss_distribution(pd, rho, lgd, ead, n_obligors,
                                 alphas=(0.999,))
    return {
        "expected_loss": out["expected_loss"],
        "credit_var_99_9": out["VaR_99.9"],
        "economic_capital": out["economic_capital_99_9"],
        "model": out["model"],
    }
