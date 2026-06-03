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

    Returns aggregate metrics + a histogram of the simulated loss distribution
    (40 bins) so the dashboard can plot the tail without re-running the sim.
    """
    out = simulate_portfolio_loss(
        pds=req.pd, eads=req.ead, lgds=req.lgd, rho=req.rho,
        n_obligors=req.n_obligors, n_sims=req.n_simulations,
        copula=req.copula, df=req.df,
    )
    losses = out.pop("losses")
    counts, edges = np.histogram(losses, bins=40)
    out["histogram_bins"] = edges.tolist()
    out["histogram_counts"] = counts.tolist()
    return PortfolioCreditResponse(**out)
