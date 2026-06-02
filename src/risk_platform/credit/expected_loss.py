"""Expected Loss + Basel IRB risk-weighted assets + portfolio aggregation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
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


def portfolio_expected_loss(
    loans: pd.DataFrame,
    pd_model: Any,
    lgd_model: Any,
    ead_col: str = "loan_amnt",
    term_col: str = "term",
) -> dict:
    """Aggregate PD/LGD/EL/RWA across a portfolio of loans.

    Parameters
    ----------
    loans     : DataFrame with the columns expected by `pd_model` and
                `lgd_model`, plus `ead_col` (defaults to `loan_amnt`).
    pd_model  : object with `predict_proba(X)` returning per-row PD.
    lgd_model : object with `predict_batch(X)` returning per-row LGD.

    Returns
    -------
    dict with:
      - per_loan       : DataFrame of (pd, lgd, ead, el, rwa) per loan
      - total_ead      : sum of EAD
      - total_el       : sum of expected loss
      - total_rwa      : sum of Basel RWA
      - weighted_pd    : EAD-weighted average PD
      - weighted_lgd   : EAD-weighted average LGD
      - el_pct_of_ead  : total_el / total_ead (%)
      - rwa_density    : total_rwa / total_ead (%)
      - top_risky      : 5 loans with highest EL
    """
    pds = np.asarray(pd_model.predict_proba(loans), dtype=float)
    if pds.ndim == 2:                        # sklearn-style
        pds = pds[:, 1] if pds.shape[1] == 2 else pds.ravel()

    lgds = np.asarray(lgd_model.predict_batch(loans), dtype=float)
    eads = loans[ead_col].astype(float).values
    terms = np.where(loans[term_col].astype(str).str.contains("60"), 60.0, 36.0) / 12.0

    els = pds * lgds * eads
    rwas = np.array(
        [basel_rwa(p, l, e, maturity_years=m)["RWA"]
         for p, l, e, m in zip(pds, lgds, eads, terms)]
    )

    per_loan = loans[[ead_col]].copy()
    per_loan["pd"] = pds
    per_loan["lgd"] = lgds
    per_loan["ead"] = eads
    per_loan["el"] = els
    per_loan["rwa"] = rwas

    total_ead = float(eads.sum())
    total_el = float(els.sum())
    total_rwa = float(rwas.sum())
    return {
        "per_loan": per_loan,
        "n_loans": int(len(loans)),
        "total_ead": total_ead,
        "total_el": total_el,
        "total_rwa": total_rwa,
        "weighted_pd": float((pds * eads).sum() / total_ead) if total_ead else 0.0,
        "weighted_lgd": float((lgds * eads).sum() / total_ead) if total_ead else 0.0,
        "el_pct_of_ead": float(total_el / total_ead * 100) if total_ead else 0.0,
        "rwa_density": float(total_rwa / total_ead * 100) if total_ead else 0.0,
        "top_risky": per_loan.nlargest(5, "el"),
    }
