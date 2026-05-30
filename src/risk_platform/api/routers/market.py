"""Market risk endpoints: /market/var, /market/es, /market/stress."""

from __future__ import annotations

from fastapi import APIRouter

from risk_platform.api.schemas import (
    MarketStressRequest, MarketStressResponse,
    MarketVaRRequest, MarketVaRResponse,
)
from risk_platform.market import MarketRisk


router = APIRouter(prefix="/market", tags=["market"])
_engine = MarketRisk()  # singleton; replaced per-request once we have real data


@router.post("/var", response_model=MarketVaRResponse)
def post_var(req: MarketVaRRequest) -> MarketVaRResponse:
    """Value at Risk by method (Historical, Parametric, MC, FHS) and alpha."""
    return MarketVaRResponse(**_engine.var(req.method, req.alpha))


@router.post("/es", response_model=MarketVaRResponse)
def post_es(req: MarketVaRRequest) -> MarketVaRResponse:
    """Expected Shortfall — conditional mean loss past VaR."""
    out = _engine.es(req.method, req.alpha)
    # Re-key ES -> VaR for schema uniformity in this stub. Phase 4 will have a
    # dedicated ESResponse schema.
    return MarketVaRResponse(VaR=out["ES"], method=out["method"],
                             alpha=out["alpha"], model=out["model"])


@router.post("/stress", response_model=MarketStressResponse)
def post_stress(req: MarketStressRequest) -> MarketStressResponse:
    """Historical stress scenario re-pricing for the bundled portfolio."""
    return MarketStressResponse(**_engine.stress(req.scenario))
