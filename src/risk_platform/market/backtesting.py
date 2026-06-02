"""Backtesting for VaR models.

Three formal tests are implemented:
  - Kupiec POF        (unconditional coverage)         chi^2(1)
  - Christoffersen    (independence + conditional cov) chi^2(1) and chi^2(2)
  - Basel traffic-light                                regulatory category

Plus rolling, out-of-sample VaR series for each method (Historical,
Parametric-Normal, Monte Carlo, FHS).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from . import var_methods as vm


# ---------------------------------------------------------------------------
# Three formal tests
# ---------------------------------------------------------------------------
def kupiec_pof(violations: np.ndarray, alpha: float = 0.05) -> dict:
    """Kupiec Proportion-of-Failures (POF) test.

    H0: exceedance rate = alpha. LR ~ chi^2(1).

    Parameters
    ----------
    violations : 0/1 array indicating per-day exceedance
    alpha      : nominal tail probability (0.05 = 95% VaR)
    """
    v = np.asarray(violations, dtype=int)
    T = len(v)
    n = int(v.sum())
    pi_hat = n / T if T else 0.0

    # Edge case: 0 or T violations — log(0) blows up. Use limiting values.
    if n == 0:
        LR = -2 * (T * np.log(1 - alpha) - 0.0)
    elif n == T:
        LR = -2 * (T * np.log(alpha) - 0.0)
    else:
        ll_h0 = n * np.log(alpha) + (T - n) * np.log(1 - alpha)
        ll_h1 = n * np.log(pi_hat) + (T - n) * np.log(1 - pi_hat)
        LR = -2 * (ll_h0 - ll_h1)

    p = 1 - chi2.cdf(LR, df=1)
    return {
        "n_obs": T,
        "exceedances": n,
        "exceedance_rate": pi_hat,
        "expected_rate": alpha,
        "LR_uc": float(LR),
        "p_value_uc": float(p),
    }


def christoffersen(violations: np.ndarray, alpha: float = 0.05) -> dict:
    """Christoffersen independence + conditional coverage tests.

    Models the violations as a 2-state Markov chain and tests whether
    transition probabilities to state 1 depend on the previous state.
    LR_ind ~ chi^2(1); LR_cc = LR_uc + LR_ind ~ chi^2(2).
    """
    v = np.asarray(violations, dtype=int)
    if len(v) < 2:
        raise ValueError("Need at least 2 observations.")

    # Transition counts n_ij = count of (v_{t-1}=i, v_t=j)
    n00 = int(np.sum((v[:-1] == 0) & (v[1:] == 0)))
    n01 = int(np.sum((v[:-1] == 0) & (v[1:] == 1)))
    n10 = int(np.sum((v[:-1] == 1) & (v[1:] == 0)))
    n11 = int(np.sum((v[:-1] == 1) & (v[1:] == 1)))

    n0 = n00 + n01  # transitions from state 0
    n1 = n10 + n11  # transitions from state 1
    total = n00 + n01 + n10 + n11
    n_violations = n01 + n11

    pi_01 = n01 / n0 if n0 else 0.0
    pi_11 = n11 / n1 if n1 else 0.0
    pi = n_violations / total if total else 0.0

    # Independence LR. Under H0: pi_01 == pi_11 == pi.
    def _safe_log(x: float) -> float:
        return np.log(x) if x > 0 else 0.0

    ll_h0_ind = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_h1_ind = (
        n00 * _safe_log(1 - pi_01) + n01 * _safe_log(pi_01) +
        n10 * _safe_log(1 - pi_11) + n11 * _safe_log(pi_11)
    )
    LR_ind = -2 * (ll_h0_ind - ll_h1_ind)

    # Conditional coverage: combine with Kupiec POF.
    uc = kupiec_pof(v, alpha)
    LR_cc = uc["LR_uc"] + LR_ind

    return {
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi_01": pi_01, "pi_11": pi_11,
        "LR_ind": float(LR_ind),
        "p_value_ind": float(1 - chi2.cdf(LR_ind, df=1)),
        "LR_cc": float(LR_cc),
        "p_value_cc": float(1 - chi2.cdf(LR_cc, df=2)),
    }


def basel_zone(exceedances: int, window: int = 250) -> str:
    """Basel traffic-light: green/yellow/red zones for 99% VaR over 250 days.

    Thresholds (per the Basel Market Risk Amendment, 1996):
      Green:  0-4  exceedances
      Yellow: 5-9  exceedances
      Red:    >=10 exceedances
    """
    if window != 250:
        warnings.warn(
            f"Basel zones are calibrated to 250-day window; got {window}. "
            "Scaling thresholds linearly — interpret with care."
        )
        scale = window / 250
        green_max = 4 * scale
        yellow_max = 9 * scale
    else:
        green_max, yellow_max = 4, 9

    if exceedances <= green_max:
        return "Green"
    if exceedances <= yellow_max:
        return "Yellow"
    return "Red"


def backtest(port_returns: pd.Series, var_series: pd.Series, alpha: float = 0.05) -> dict:
    """Run all three tests on an aligned (returns, VaR) pair.

    Inputs must be the *out-of-sample* portion: VaR_t was forecast at t-1
    using info up through t-1, realized return is r_t.
    """
    common = port_returns.index.intersection(var_series.index)
    r = port_returns.loc[common]
    v = var_series.loc[common]
    violations = (r < -v).astype(int).values

    uc = kupiec_pof(violations, alpha)
    cc = christoffersen(violations, alpha)
    # Basel uses the most-recent 250-day window.
    recent_excs = int(violations[-250:].sum()) if len(violations) >= 250 else int(violations.sum())
    zone = basel_zone(recent_excs, window=min(250, len(violations)))

    return {**uc, **cc, "basel_zone": zone, "recent_250d_exc": recent_excs}


# ---------------------------------------------------------------------------
# Rolling, out-of-sample VaR series
# ---------------------------------------------------------------------------
def rolling_historical_var(
    port: pd.Series, alpha: float = 0.05, window: int = 500
) -> pd.Series:
    """Rolling empirical-quantile VaR. Predicts VaR_t using r_{t-window..t-1}."""
    # rolling().quantile uses the trailing window INCLUDING the current obs;
    # we want STRICTLY past data, so shift by 1.
    return -port.shift(1).rolling(window).quantile(alpha).dropna()


def rolling_parametric_normal_var(
    port: pd.Series, alpha: float = 0.05, window: int = 500
) -> pd.Series:
    """Rolling Parametric-Normal VaR: VaR_t = -(mu_{t-1} + sigma_{t-1} * Phi^-1(alpha))."""
    mu = port.shift(1).rolling(window).mean()
    sd = port.shift(1).rolling(window).std()
    z = norm.ppf(alpha)
    return -(mu + sd * z).dropna()


def rolling_monte_carlo_var(
    port: pd.Series,
    alpha: float = 0.05,
    window: int = 500,
    n_sims: int = 5_000,
    seed: int = 42,
) -> pd.Series:
    """Rolling Monte Carlo (univariate Normal) VaR.

    For a univariate portfolio this is the same model as Parametric Normal —
    we include it for backtesting completeness (verifies our simulator).
    """
    rng = np.random.default_rng(seed)
    out = pd.Series(index=port.index, dtype=float)
    vals = port.values
    for t in range(window, len(port)):
        w = vals[t - window:t]
        sims = rng.normal(w.mean(), w.std(ddof=1), size=n_sims)
        out.iloc[t] = -np.quantile(sims, alpha)
    return out.dropna()


def rolling_fhs_var(
    port: pd.Series,
    alpha: float = 0.05,
    window: int = 1_000,
    refit_every: int = 60,
    n_sims: int = 5_000,
    seed: int = 42,
) -> pd.Series:
    """Rolling Filtered Historical Simulation VaR.

    Refits GARCH every `refit_every` days on a trailing window of `window`
    days. Between refits, uses the GARCH recursion to update sigma daily.
    """
    rng = np.random.default_rng(seed)
    out = pd.Series(index=port.index, dtype=float)
    vals = port.values

    cur_omega = cur_alpha = cur_beta = cur_mu = cur_sigma_sq = None
    cur_resids = None
    days_since_refit = refit_every  # force initial fit

    for t in range(window, len(port)):
        if days_since_refit >= refit_every:
            train = port.iloc[t - window:t]
            g = vm.fit_garch(train)
            res = g["result"]
            cur_mu = float(res.params["mu"]) / 100.0
            cur_omega = float(res.params["omega"]) / 10_000.0
            cur_alpha = float(res.params["alpha[1]"])
            cur_beta = float(res.params["beta[1]"])
            cur_sigma_sq = float(g["sigma_t"].iloc[-1]) ** 2
            cur_resids = g["std_resid"].values
            days_since_refit = 0
        else:
            # GARCH recursion using yesterday's observation
            a_prev = vals[t - 1] - cur_mu
            cur_sigma_sq = cur_omega + cur_alpha * a_prev**2 + cur_beta * cur_sigma_sq

        sigma_fc = np.sqrt(cur_sigma_sq)
        z_samp = rng.choice(cur_resids, size=n_sims, replace=True)
        r_sims = cur_mu + sigma_fc * z_samp
        out.iloc[t] = -np.quantile(r_sims, alpha)
        days_since_refit += 1

    return out.dropna()


# ---------------------------------------------------------------------------
# Convenience: build a results table across all methods × alphas
# ---------------------------------------------------------------------------
def backtest_all_methods(
    port: pd.Series,
    alpha: float = 0.05,
    window: int = 500,
    fhs_window: int = 1_000,
    refit_every: int = 60,
    n_sims: int = 5_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute rolling VaR + backtest stats for all 4 methods at given alpha."""
    methods = {
        "Historical": rolling_historical_var(port, alpha, window),
        "Parametric (Normal)": rolling_parametric_normal_var(port, alpha, window),
        "Monte Carlo (Normal)": rolling_monte_carlo_var(port, alpha, window, n_sims, seed),
        "FHS (GARCH(1,1))": rolling_fhs_var(port, alpha, fhs_window, refit_every, n_sims, seed),
    }
    rows = []
    for name, var_s in methods.items():
        bt = backtest(port, var_s, alpha)
        rows.append({
            "method": name,
            "alpha": alpha,
            "confidence": f"{int((1-alpha)*100)}%",
            "n_obs": bt["n_obs"],
            "exceedances": bt["exceedances"],
            "exc_rate_%": round(bt["exceedance_rate"] * 100, 3),
            "expected_%": round(alpha * 100, 3),
            "Kupiec_p": round(bt["p_value_uc"], 4),
            "Christof_ind_p": round(bt["p_value_ind"], 4),
            "Christof_cc_p": round(bt["p_value_cc"], 4),
            "Basel_zone": bt["basel_zone"],
            "recent_250d_exc": bt["recent_250d_exc"],
        })
    return pd.DataFrame(rows), methods
