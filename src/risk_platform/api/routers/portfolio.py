"""Portfolio credit risk endpoints: /portfolio/credit_var."""

from __future__ import annotations

from fastapi import APIRouter

from risk_platform.api.schemas import (
    PortfolioCreditRequest, PortfolioCreditResponse,
)
from risk_platform.portfolio import simulate_portfolio_losses


router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/credit_var", response_model=PortfolioCreditResponse)
def post_credit_var(req: PortfolioCreditRequest) -> PortfolioCreditResponse:
    """Simulated portfolio loss distribution under Gaussian or t-copula."""
    out = simulate_portfolio_losses(
        pd=req.pd, rho=req.rho, n_obligors=req.n_obligors,
        n_simulations=req.n_simulations, copula=req.copula, df=req.df,
    )
    return PortfolioCreditResponse(**out)
