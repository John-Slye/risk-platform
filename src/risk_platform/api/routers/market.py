"""Market risk endpoints: /market/var, /market/es, /market/stress.

Accept an optional `portfolio` (tickers + weights) in the request body. If
omitted, the default 9-asset portfolio is used. Per-portfolio MarketRisk
instances are cached in-process so repeated requests for the same portfolio
are instant.
"""

from __future__ import annotations

from fastapi import APIRouter

from risk_platform.api.schemas import (
    MarketStressRequest, MarketStressResponse,
    MarketVaRRequest, MarketVaRResponse,
)
from risk_platform.market import get_market_risk


router = APIRouter(prefix="/market", tags=["market"])


def _engine_from_request(req) -> object:
    """Resolve the MarketRisk instance for the request, hitting cache if hot."""
    pf = req.portfolio
    if pf is None:
        return get_market_risk()
    return get_market_risk(tickers=pf.tickers, weights=pf.weights)


@router.post("/var", response_model=MarketVaRResponse)
def post_var(req: MarketVaRRequest) -> MarketVaRResponse:
    """Value at Risk by method (Historical, Parametric, MC, FHS) and alpha."""
    engine = _engine_from_request(req)
    return MarketVaRResponse(**engine.var(req.method, req.alpha))


@router.post("/es", response_model=MarketVaRResponse)
def post_es(req: MarketVaRRequest) -> MarketVaRResponse:
    """Expected Shortfall — conditional mean loss past VaR."""
    engine = _engine_from_request(req)
    out = engine.es(req.method, req.alpha)
    return MarketVaRResponse(VaR=out["ES"], method=out["method"],
                             alpha=out["alpha"], model=out["model"])


@router.post("/stress", response_model=MarketStressResponse)
def post_stress(req: MarketStressRequest) -> MarketStressResponse:
    """Historical stress scenario re-pricing for the chosen portfolio."""
    engine = _engine_from_request(req)
    return MarketStressResponse(**engine.stress(req.scenario))
