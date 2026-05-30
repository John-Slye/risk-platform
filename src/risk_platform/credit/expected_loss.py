"""Expected Loss and Basel IRB risk-weighted assets, stubs for Phase 0."""

from __future__ import annotations

import math

from scipy.stats import norm


def expected_loss(pd: float, lgd: float, ead: float) -> float:
    """EL = PD * LGD * EAD. The simplest formula in credit risk."""
    return pd * lgd * ead


def basel_rwa(pd: float, lgd: float, ead: float, maturity_years: float = 1.0,
              asset_class: str = "corporate") -> dict:
    """Basel III Internal Ratings-Based (IRB) risk-weighted asset.

    Phase 0 ships the corporate IRB formula directly (it is simple enough to
    write from scratch). Phase 2 will add retail and sovereign asset classes.
    """
    # Asset correlation R for corporate exposures (Basel III formula).
    R = 0.12 * (1 - math.exp(-50 * pd)) / (1 - math.exp(-50)) + \
        0.24 * (1 - (1 - math.exp(-50 * pd)) / (1 - math.exp(-50)))
    # Maturity adjustment.
    b = (0.11852 - 0.05478 * math.log(max(pd, 1e-6))) ** 2
    M_adj = (1 + (maturity_years - 2.5) * b) / (1 - 1.5 * b)
    # Conditional PD under the supervisory worst-case (99.9% factor draw).
    cond_pd = norm.cdf(
        (norm.ppf(pd) + math.sqrt(R) * norm.ppf(0.999)) / math.sqrt(1 - R)
    )
    k = lgd * (cond_pd - pd) * M_adj
    rwa = k * 12.5 * ead
    return {
        "K": k,
        "RWA": rwa,
        "asset_correlation": R,
        "maturity_adjustment": M_adj,
        "conditional_pd": cond_pd,
    }
