"""Joint-default Monte Carlo via one-factor Gaussian and Student-t copulas.

Vasicek-style factor structure (same as `vasicek.py`):
    A_i = sqrt(rho) * M + sqrt(1-rho) * eps_i

For the Gaussian copula: M, eps_i ~ N(0, 1).
For the t-copula:        scale the Gaussian asset value by sqrt(chi2_df / df)
                         to give fatter joint tails.

The point of doing both: in the Gaussian copula, extreme co-defaults are
exponentially rare; in the t-copula they happen at non-trivial frequency.
That difference is the entire David X. Li / 2008 mortgage-tranche story.
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t


def _simulate_factor_returns(
    n_obligors: int, n_sims: int, rho: float,
    copula: Literal["gaussian", "t"], df: int, rng: np.random.Generator,
) -> np.ndarray:
    """Simulate `n_sims` x `n_obligors` matrix of latent asset returns A_i."""
    sqrt_rho = np.sqrt(rho)
    sqrt_1m = np.sqrt(1 - rho)
    M = rng.normal(size=n_sims)                  # systematic factor per sim
    eps = rng.normal(size=(n_sims, n_obligors))  # idiosyncratic per obligor x sim
    Z = sqrt_rho * M[:, None] + sqrt_1m * eps    # Gaussian latent asset value

    if copula == "gaussian":
        return Z
    if copula == "t":
        # Scale by 1 / sqrt(W / df) where W ~ chi2(df); same W across obligors
        # within a sim preserves joint tail dependence.
        W = rng.chisquare(df, size=n_sims) / df
        return Z / np.sqrt(W)[:, None]
    raise ValueError(f"copula must be 'gaussian' or 't', got {copula!r}")


def _default_thresholds(
    pds: np.ndarray, copula: Literal["gaussian", "t"], df: int,
) -> np.ndarray:
    """Per-obligor default thresholds in the latent-return distribution."""
    if copula == "gaussian":
        return norm.ppf(pds)
    return student_t.ppf(pds, df)


def simulate_portfolio_loss(
    pds: float | np.ndarray = 0.05,
    eads: float | np.ndarray = 1.0,
    lgds: float | np.ndarray = 0.45,
    rho: float = 0.15,
    n_obligors: int = 1_000,
    n_sims: int = 50_000,
    copula: Literal["gaussian", "t"] = "gaussian",
    df: int = 5,
    seed: Optional[int] = 42,
) -> dict:
    """Simulate the portfolio loss distribution under a one-factor copula.

    Scalar PD/EAD/LGD => homogeneous portfolio. Array PD/EAD/LGD =>
    heterogeneous portfolio (length must equal n_obligors).

    Returns dict with: losses (np.ndarray of length n_sims), and risk metrics
    (expected_loss, credit_var_99, credit_var_99_9, economic_capital,
    tail_es_99_9, hhi).
    """
    rng = np.random.default_rng(seed)

    # Broadcast PDs / EADs / LGDs to length n_obligors.
    pds = np.broadcast_to(np.asarray(pds, dtype=float), (n_obligors,)).copy()
    eads = np.broadcast_to(np.asarray(eads, dtype=float), (n_obligors,)).copy()
    lgds = np.broadcast_to(np.asarray(lgds, dtype=float), (n_obligors,)).copy()

    A = _simulate_factor_returns(n_obligors, n_sims, rho, copula, df, rng)
    thresholds = _default_thresholds(pds, copula, df)

    defaults = (A < thresholds[None, :])       # shape (n_sims, n_obligors)
    loss_per_obligor = eads * lgds             # length n_obligors
    losses = (defaults * loss_per_obligor[None, :]).sum(axis=1)  # length n_sims

    total_ead = float(eads.sum())
    el = float((pds * lgds * eads).sum())          # expected, analytical
    var_99 = float(np.quantile(losses, 0.99))
    var_99_9 = float(np.quantile(losses, 0.999))
    tail = losses[losses >= var_99_9]
    tail_es = float(tail.mean()) if len(tail) else float("nan")
    # Herfindahl on EAD shares (concentration measure)
    shares = eads / total_ead if total_ead else np.zeros_like(eads)
    hhi = float((shares ** 2).sum())

    return {
        "copula": copula,
        "df": df if copula == "t" else None,
        "n_obligors": n_obligors,
        "n_simulations": n_sims,
        "total_ead": total_ead,
        "expected_loss": el,
        "credit_var_99": var_99,
        "credit_var_99_9": var_99_9,
        "economic_capital": var_99_9 - el,
        "tail_es_99_9": tail_es,
        "hhi": hhi,
        "model": f"copula-{copula}-0.1.0",
        "losses": losses,    # full distribution for plotting; can be discarded
    }


# Backward-compat shim used by the API endpoint.
def simulate_portfolio_losses(
    pd_rate: float = 0.05, rho: float = 0.15, n_obligors: int = 1_000,
    lgd: float = 0.92, ead: float = 15_000.0,
    n_simulations: int = 10_000, copula: str = "gaussian", df: int = 5,
) -> dict:
    out = simulate_portfolio_loss(
        pds=pd_rate, eads=ead, lgds=lgd, rho=rho,
        n_obligors=n_obligors, n_sims=n_simulations,
        copula=copula, df=df,
    )
    out.pop("losses", None)   # don't return the giant array to the API
    return out
