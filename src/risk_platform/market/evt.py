"""Extreme Value Theory — Peaks-Over-Threshold (POT) with Generalized Pareto.

Models only the tail of the loss distribution rather than the whole thing.
Standard reference: McNeil, Frey & Embrechts, "Quantitative Risk Management."

The Pickands-Balkema-de Haan theorem says: for a wide class of underlying
distributions, the conditional distribution of excesses over a high threshold
u converges to a Generalized Pareto Distribution (GPD) as u rises. So:

  1. Pick threshold u (e.g., 95th percentile of losses).
  2. Form excesses Y_i = X_i - u for X_i > u.
  3. Fit GPD(xi, sigma) to {Y_i} by MLE.
  4. VaR_alpha = u + (sigma/xi) * (((n/n_u) * alpha)^(-xi) - 1)
     ES_alpha  = VaR_alpha / (1-xi) + (sigma - xi*u)/(1-xi)     [if xi < 1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import genpareto


def fit_pot(
    returns: pd.Series, threshold_pct: float = 0.95
) -> dict:
    """Fit GPD to losses exceeding a high threshold.

    Parameters
    ----------
    returns       : daily portfolio returns (decimal). Losses are -returns.
    threshold_pct : threshold as a quantile of LOSSES. Default 0.95 means
                    "treat the top 5% of losses as tail observations."

    Returns
    -------
    dict with:
      u         : threshold (positive loss number)
      xi        : GPD shape parameter (positive = heavy tail)
      sigma     : GPD scale parameter
      n         : total observations
      n_u       : number of exceedances
      zeta_u    : empirical P(loss > u) = n_u / n
      excesses  : the actual excess values (losses - u, sorted ascending)
    """
    losses = -returns.values
    losses = losses[~np.isnan(losses)]
    n = len(losses)
    u = float(np.quantile(losses, threshold_pct))
    excesses = losses[losses > u] - u
    n_u = len(excesses)

    if n_u < 30:
        raise ValueError(
            f"Only {n_u} exceedances above threshold. Need >= 30 for stable GPD. "
            "Try a lower threshold_pct (e.g. 0.93)."
        )

    # Fit GPD by MLE. scipy returns (xi, loc, sigma). We force loc=0 because
    # excesses are by construction >= 0 (we already subtracted u).
    xi, _loc, sigma = genpareto.fit(excesses, floc=0)
    return {
        "u": u,
        "xi": float(xi),
        "sigma": float(sigma),
        "n": int(n),
        "n_u": int(n_u),
        "zeta_u": n_u / n,
        "excesses": np.sort(excesses),
    }


def evt_var(returns: pd.Series, alpha: float = 0.01, threshold_pct: float = 0.95) -> float:
    """EVT VaR at tail probability `alpha` (e.g. 0.01 for 99%, 0.005 for 99.5%).

    `threshold_pct` must be <= 1 - alpha (i.e. threshold must lie below the
    quantile we want to estimate); otherwise the formula does not apply.
    """
    fit = fit_pot(returns, threshold_pct)
    xi, sigma, u = fit["xi"], fit["sigma"], fit["u"]
    n, n_u = fit["n"], fit["n_u"]

    if alpha > 1 - threshold_pct:
        raise ValueError(
            f"alpha={alpha} requires threshold below the {1-alpha:.4f} quantile, "
            f"but threshold is at the {threshold_pct} quantile. "
            "Lower the threshold or raise the confidence."
        )

    ratio = (n / n_u) * alpha
    if abs(xi) > 1e-8:
        var = u + (sigma / xi) * (ratio ** (-xi) - 1)
    else:
        var = u - sigma * np.log(ratio)
    return float(var)


def evt_es(returns: pd.Series, alpha: float = 0.01, threshold_pct: float = 0.95) -> float:
    """EVT Expected Shortfall under GPD tail. Defined only if xi < 1."""
    fit = fit_pot(returns, threshold_pct)
    xi, sigma, u = fit["xi"], fit["sigma"], fit["u"]
    if xi >= 1:
        return float("nan")
    var = evt_var(returns, alpha, threshold_pct)
    return float(var / (1 - xi) + (sigma - xi * u) / (1 - xi))


def evt_summary(
    returns: pd.Series,
    alphas: tuple[float, ...] = (0.05, 0.01, 0.005, 0.001),
    threshold_pct: float = 0.95,
) -> pd.DataFrame:
    """VaR and ES across confidence levels under one threshold choice."""
    fit = fit_pot(returns, threshold_pct)
    rows = []
    for alpha in alphas:
        if alpha > 1 - threshold_pct:
            rows.append({
                "alpha": alpha, "confidence": f"{(1-alpha)*100:.1f}%",
                "VaR_%": np.nan, "ES_%": np.nan,
                "note": f"alpha > 1 - threshold ({threshold_pct}); not applicable",
            })
            continue
        rows.append({
            "alpha": alpha,
            "confidence": f"{(1-alpha)*100:.2f}%".rstrip("0").rstrip("."),
            "VaR_%": evt_var(returns, alpha, threshold_pct) * 100,
            "ES_%":  evt_es (returns, alpha, threshold_pct) * 100,
            "note": "",
        })
    out = pd.DataFrame(rows)
    out.attrs["fit"] = fit
    return out


def mean_excess(losses: np.ndarray, n_points: int = 50) -> pd.DataFrame:
    """Mean Residual Life function for threshold selection.

    The mean excess function e(u) = E[X - u | X > u]. Under a GPD tail with
    xi < 1, e(u) is LINEAR in u with slope xi/(1-xi). So picking the threshold
    where the empirical e(u) becomes approximately linear is a defensible choice.
    """
    losses = losses[~np.isnan(losses)]
    qs = np.linspace(0.50, 0.995, n_points)
    rows = []
    for q in qs:
        u = float(np.quantile(losses, q))
        tail = losses[losses > u]
        if len(tail) < 10:
            continue
        rows.append({
            "threshold_pct": q,
            "threshold": u,
            "mean_excess": float((tail - u).mean()),
            "n_exceedances": int(len(tail)),
        })
    return pd.DataFrame(rows)
