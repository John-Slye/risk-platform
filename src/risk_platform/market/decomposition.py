"""Component and Marginal VaR — Euler decomposition.

Given a portfolio with weights w and an asset covariance matrix Sigma,
the Parametric-Normal portfolio VaR is:

    VaR_p  =  -mu_p  +  z_alpha * sigma_p           where sigma_p = sqrt(w' Sigma w)

The risk decomposition is:

    Marginal VaR_i   =  d VaR_p / d w_i        =  z_alpha * (Sigma w)_i / sigma_p
    Component VaR_i  =  w_i * Marginal VaR_i

By Euler's theorem on homogeneous functions of degree 1,

    sum_i (Component VaR_i)  =  z_alpha * sigma_p  =  VaR_p (ignoring mean)

so component VaRs literally add up to the portfolio VaR — they tell you
which assets the risk *actually* comes from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def component_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
    include_mean: bool = False,
) -> pd.DataFrame:
    """Per-asset risk contributions to total portfolio VaR.

    Parameters
    ----------
    returns      : T x N asset returns (decimal). Used to estimate Sigma and mu.
    weights      : length-N Series indexed by asset, summing to 1.
    alpha        : tail probability for VaR (0.05 = 95%, 0.01 = 99%).
    include_mean : if True, includes -mu_p in the VaR formula; usually False at
                   daily horizon where mu << sigma.

    Returns
    -------
    DataFrame indexed by asset with columns:
      - weight
      - marginal_var          : sensitivity of VaR to a 1-unit weight change
      - component_var         : asset's contribution to total VaR
      - pct_contribution      : component_var / total_var, in percent

    Plus a 'TOTAL' summary row. The sum of component_var across assets
    equals the portfolio Parametric-Normal VaR (Euler decomposition check).
    """
    w = weights.reindex(returns.columns).fillna(0.0).values
    cov = returns.cov().values
    mu = returns.mean().values

    sigma_p = float(np.sqrt(w @ cov @ w))
    z = -norm.ppf(alpha)              # positive: ~1.645 for 95%, ~2.326 for 99%
    mu_p = float(w @ mu)
    var_p = z * sigma_p - (mu_p if include_mean else 0.0)

    marginal_var = z * (cov @ w) / sigma_p
    comp_var = w * marginal_var
    pct = comp_var / var_p * 100.0 if var_p else np.zeros_like(comp_var)

    df = pd.DataFrame({
        "weight": w,
        "marginal_var": marginal_var,
        "component_var": comp_var,
        "pct_contribution": pct,
    }, index=returns.columns)

    # Sanity row that confirms Euler decomposition holds.
    df.loc["TOTAL"] = [
        df["weight"].sum(),
        np.nan,
        df["component_var"].sum(),
        df["pct_contribution"].sum(),
    ]
    return df


def euler_check(returns: pd.DataFrame, weights: pd.Series, alpha: float = 0.05) -> dict:
    """Verify sum(component_var) ≈ Parametric-Normal portfolio VaR (sanity)."""
    w = weights.reindex(returns.columns).fillna(0.0).values
    cov = returns.cov().values
    sigma_p = float(np.sqrt(w @ cov @ w))
    z = -norm.ppf(alpha)
    var_direct = z * sigma_p
    df = component_var(returns, weights, alpha)
    var_sum = df.loc[returns.columns, "component_var"].sum()
    return {
        "var_direct": var_direct,
        "sum_component_var": float(var_sum),
        "abs_error": abs(var_direct - var_sum),
        "relative_error": abs(var_direct - var_sum) / var_direct,
    }
