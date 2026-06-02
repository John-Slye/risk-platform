"""Value-at-Risk and Expected Shortfall — three method families.

A VaR / ES result is always reported as a **positive loss number**:
"1-day 95% VaR = 0.012" means "there is a 5% chance of losing more
than 1.2% of portfolio value over the next day."

Methods implemented here:
  - Historical Simulation       (model-free, empirical quantile)
  - Variance-Covariance         (closed-form, Normal or Student-t)
  - Monte Carlo                 (simulate from MV-Normal or MV-t)
  - Filtered Historical Sim     (GARCH(1,1)-filtered residuals, resampled)

All functions accept:
  - `alpha`: tail probability in (0, 1). Use 0.05 for 95% VaR, 0.01 for 99%.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm, t

try:
    from arch import arch_model  # type: ignore
except ImportError:  # pragma: no cover
    arch_model = None  # FHS will raise a clear error if called without `arch`


# ---------------------------------------------------------------------------
# Historical Simulation
# ---------------------------------------------------------------------------
def historical_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """Empirical alpha-quantile loss of `returns`.

    No distributional assumption — the empirical CDF IS our model. Strength:
    captures whatever fat tails / skew exist in your sample. Weakness: assumes
    the future looks like the past, and the worst possible VaR is the worst
    historical loss (it can't extrapolate).
    """
    _check_alpha(alpha)
    return float(-np.quantile(returns, alpha))


def historical_es(returns: pd.Series, alpha: float = 0.05) -> float:
    """Expected Shortfall (CVaR) = mean loss conditional on loss exceeding VaR.

    Computed as the average of returns in the lower-alpha tail. Always reported
    as a positive loss number.
    """
    _check_alpha(alpha)
    threshold = np.quantile(returns, alpha)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


# ---------------------------------------------------------------------------
# Parametric (Variance-Covariance)
# ---------------------------------------------------------------------------
def parametric_var(
    returns: pd.Series, alpha: float = 0.05, dist: str = "normal"
) -> float:
    """Closed-form VaR under a Normal or Student-t assumption.

    `dist="normal"`: VaR = -(mu + sigma * Phi^-1(alpha)).
    `dist="t"`:      Fits a location-scale Student-t (df, loc, scale) by MLE
                     and reads off the quantile. Captures fat tails.
    """
    _check_alpha(alpha)
    if dist == "normal":
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        return float(-(mu + sigma * norm.ppf(alpha)))
    if dist == "t":
        df, loc, scale = t.fit(returns)
        return float(-t.ppf(alpha, df, loc=loc, scale=scale))
    raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")


def parametric_es(
    returns: pd.Series, alpha: float = 0.05, dist: str = "normal"
) -> float:
    """Closed-form Expected Shortfall under Normal or Student-t.

    Normal: ES = -mu + sigma * phi(z) / alpha,  where z = Phi^-1(alpha).
    Student-t (location-scale): ES = -loc + scale * tau(z)/alpha * (df + z^2)/(df - 1),
                                where z = T_df^-1(alpha) (standardized) and tau
                                is the standardized t PDF.
    """
    _check_alpha(alpha)
    if dist == "normal":
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        z = norm.ppf(alpha)
        return float(-mu + sigma * norm.pdf(z) / alpha)
    if dist == "t":
        df, loc, scale = t.fit(returns)
        if df <= 1:
            return float("nan")  # ES undefined for df <= 1
        z = t.ppf(alpha, df)  # standardized quantile
        es_std = (t.pdf(z, df) / alpha) * (df + z**2) / (df - 1)
        return float(-loc + scale * es_std)
    raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def monte_carlo_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10_000,
    dist: str = "normal",
    seed: int | None = 42,
) -> float:
    """Monte Carlo VaR for a portfolio with weights `weights`.

    `returns` is the asset-level DataFrame (T x N), `weights` a Series indexed
    by asset (length N, summing to 1).

    For `dist="normal"`: draws joint asset returns from multivariate normal
    fit to the sample mean vector and covariance matrix.

    For `dist="t"`: draws from multivariate Student-t. We fit the df by
    matching it to the portfolio-return univariate t-fit (a common shortcut
    that's exact only when assets are symmetric and tail dependence is mild,
    but works well in practice).
    """
    sims = _simulate_returns(returns, weights, n_sims, dist, seed)
    port_sims = sims @ weights.values
    return float(-np.quantile(port_sims, alpha))


def monte_carlo_es(
    returns: pd.DataFrame,
    weights: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10_000,
    dist: str = "normal",
    seed: int | None = 42,
) -> float:
    """Monte Carlo Expected Shortfall — mean of simulated losses below VaR."""
    sims = _simulate_returns(returns, weights, n_sims, dist, seed)
    port_sims = sims @ weights.values
    threshold = np.quantile(port_sims, alpha)
    return float(-port_sims[port_sims <= threshold].mean())


# ---------------------------------------------------------------------------
# Filtered Historical Simulation (FHS)
# ---------------------------------------------------------------------------
def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> dict:
    """Fit a GARCH(p, q) to a return series and return all the pieces FHS needs.

    Returns are passed to `arch` after multiplying by 100 (percent units) for
    numerical stability — GARCH optimization is notoriously fussy with very
    small numbers. We rescale back to decimals on the way out.

    Returns dict with:
      - mu                 : fitted constant mean (decimal units)
      - sigma_t            : pd.Series of in-sample conditional vol (decimal)
      - std_resid          : pd.Series of standardized residuals (dimensionless)
      - sigma_forecast     : scalar one-step-ahead conditional vol (decimal)
      - result             : underlying arch fit result (for diagnostics)
    """
    if arch_model is None:
        raise ImportError("FHS requires `arch`. Install with `pip install arch`.")

    x = returns.dropna() * 100.0  # percent units
    model = arch_model(
        x,
        mean="Constant",
        vol="GARCH",
        p=p,
        q=q,
        dist=dist,
        rescale=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp="off", show_warning=False)

    # In-sample conditional vol (percent units), shift to decimals.
    sigma_pct = res.conditional_volatility
    sigma_t = sigma_pct / 100.0

    # Standardized residuals are scale-free (numerator and denominator both in %).
    std_resid = (x - res.params["mu"]) / sigma_pct

    # One-step-ahead forecast of sigma (percent units → decimal).
    fc = res.forecast(horizon=1, reindex=False)
    sigma_fc_pct = float(np.sqrt(fc.variance.values[-1, 0]))
    sigma_forecast = sigma_fc_pct / 100.0

    mu_dec = float(res.params["mu"]) / 100.0

    return {
        "mu": mu_dec,
        "sigma_t": sigma_t,
        "std_resid": std_resid,
        "sigma_forecast": sigma_forecast,
        "result": res,
    }


def filtered_historical_var(
    returns: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10_000,
    seed: int | None = 42,
    garch: dict | None = None,
) -> float:
    """FHS VaR: GARCH-filtered residuals resampled and rescaled by current sigma.

    Steps:
      1. Fit GARCH(1,1) to `returns`.
      2. Extract standardized residuals z_t = (r_t - mu) / sigma_t.
      3. Get one-step-ahead conditional vol sigma_{T+1}.
      4. Resample z's with replacement, build next-day returns
         r_sim = mu + sigma_{T+1} * z_sampled.
      5. VaR = -alpha-quantile of simulated returns.

    Pass `garch=fit_garch(returns)` if you've already fit it (the rolling-window
    backtest in Phase 3 will use this to avoid re-fitting).
    """
    _check_alpha(alpha)
    g = garch if garch is not None else fit_garch(returns)
    rng = np.random.default_rng(seed)
    z = g["std_resid"].values
    z_sampled = rng.choice(z, size=n_sims, replace=True)
    r_sim = g["mu"] + g["sigma_forecast"] * z_sampled
    return float(-np.quantile(r_sim, alpha))


def filtered_historical_es(
    returns: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 10_000,
    seed: int | None = 42,
    garch: dict | None = None,
) -> float:
    """Expected Shortfall from FHS — mean of simulated losses past VaR."""
    _check_alpha(alpha)
    g = garch if garch is not None else fit_garch(returns)
    rng = np.random.default_rng(seed)
    z = g["std_resid"].values
    z_sampled = rng.choice(z, size=n_sims, replace=True)
    r_sim = g["mu"] + g["sigma_forecast"] * z_sampled
    threshold = np.quantile(r_sim, alpha)
    return float(-r_sim[r_sim <= threshold].mean())


# ---------------------------------------------------------------------------
# Convenience: produce a results table across all methods × confidence levels
# ---------------------------------------------------------------------------
def var_summary(
    returns: pd.DataFrame,
    weights: pd.Series,
    alphas: tuple[float, ...] = (0.05, 0.01),
    n_sims: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """One-stop comparison: VaR + ES for every method at every alpha."""
    port = returns @ weights
    # Fit GARCH once and reuse for every alpha (FHS only differs in the quantile).
    garch = fit_garch(port) if arch_model is not None else None
    rows = []
    for alpha in alphas:
        rows.append(_row("Historical", alpha,
                         historical_var(port, alpha),
                         historical_es(port, alpha)))
        rows.append(_row("Parametric (normal)", alpha,
                         parametric_var(port, alpha, "normal"),
                         parametric_es(port, alpha, "normal")))
        rows.append(_row("Parametric (t)", alpha,
                         parametric_var(port, alpha, "t"),
                         parametric_es(port, alpha, "t")))
        rows.append(_row("Monte Carlo (normal)", alpha,
                         monte_carlo_var(returns, weights, alpha, n_sims, "normal", seed),
                         monte_carlo_es(returns, weights, alpha, n_sims, "normal", seed)))
        rows.append(_row("Monte Carlo (t)", alpha,
                         monte_carlo_var(returns, weights, alpha, n_sims, "t", seed),
                         monte_carlo_es(returns, weights, alpha, n_sims, "t", seed)))
        if garch is not None:
            rows.append(_row("FHS (GARCH(1,1))", alpha,
                             filtered_historical_var(port, alpha, n_sims, seed, garch=garch),
                             filtered_historical_es (port, alpha, n_sims, seed, garch=garch)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _simulate_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    n_sims: int,
    dist: str,
    seed: int | None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = returns.mean().values
    cov = returns.cov().values

    if dist == "normal":
        return rng.multivariate_normal(mu, cov, size=n_sims)

    if dist == "t":
        # Multivariate t = MV-Normal scaled by sqrt(df / chi2_df).
        # We fit df on the *portfolio* returns (univariate) — a standard shortcut.
        port = returns @ weights
        df, _, _ = t.fit(port)
        if df <= 2:
            df = 3.0  # cov is undefined for df <= 2; floor to a usable value
        z = rng.multivariate_normal(np.zeros_like(mu), cov, size=n_sims)
        g = rng.chisquare(df, size=n_sims) / df
        return mu + z / np.sqrt(g)[:, None]

    raise ValueError(f"dist must be 'normal' or 't', got {dist!r}")


def _row(method: str, alpha: float, var: float, es: float) -> dict:
    return {
        "method": method,
        "alpha": alpha,
        "confidence": f"{int((1 - alpha) * 100)}%",
        "VaR": var,
        "ES": es,
    }
