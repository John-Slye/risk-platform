"""Portfolio credit risk endpoints: /portfolio/credit_var."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from risk_platform.api.schemas import (
    PortfolioCreditRequest, PortfolioCreditResponse,
)
from risk_platform.portfolio.copula import simulate_portfolio_loss


router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/credit_var", response_model=PortfolioCreditResponse)
def post_credit_var(req: PortfolioCreditRequest) -> PortfolioCreditResponse:
    """Simulated portfolio loss distribution under Gaussian or t-copula.

    If `obligors` is provided, runs heterogeneous PD/LGD/EAD per obligor.
    Otherwise runs homogeneous parameters (pd / lgd / ead / n_obligors).

    Returns aggregate metrics + a histogram of the simulated loss
    distribution (40 bins) so the dashboard can plot the tail without
    re-running the simulation.
    """
    if req.obligors:
        pds = np.array([o.pd for o in req.obligors])
        lgds = np.array([o.lgd for o in req.obligors])
        eads = np.array([o.ead for o in req.obligors])
        n_obligors = len(req.obligors)
    else:
        pds = req.pd
        lgds = req.lgd
        eads = req.ead
        n_obligors = req.n_obligors

    out = simulate_portfolio_loss(
        pds=pds, eads=eads, lgds=lgds, rho=req.rho,
        n_obligors=n_obligors, n_sims=req.n_simulations,
        copula=req.copula, df=req.df,
    )
    losses = out.pop("losses")
    counts, edges = np.histogram(losses, bins=40)
    out["histogram_bins"] = edges.tolist()
    out["histogram_counts"] = counts.tolist()
    return PortfolioCreditResponse(**out)
