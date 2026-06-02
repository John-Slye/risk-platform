"""Unified market + credit risk report: /risk_report.

Reuses the credit router's loaded scorecard/lgd singletons so the trained
pickles aren't re-loaded.
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from risk_platform.api.schemas import (
    ExpectedLossResponse, MarketVaRResponse, PortfolioCreditResponse,
    RiskReportRequest, RiskReportResponse,
)
from risk_platform.api.routers.credit import _lgd, _scorecard
from risk_platform.credit import basel_rwa, expected_loss
from risk_platform.market import MarketRisk
from risk_platform.portfolio import simulate_portfolio_losses


router = APIRouter(prefix="", tags=["report"])


@router.post("/risk_report", response_model=RiskReportResponse)
def post_risk_report(req: RiskReportRequest) -> RiskReportResponse:
    """Run market VaR + loan EL + portfolio Credit VaR and return all three."""
    # Market
    mr = MarketRisk().var(req.market_method, req.market_alpha)
    market_resp = MarketVaRResponse(**mr)

    # Credit (loan-level) using shared singletons
    X = pd.DataFrame([req.loan.to_model_dict()])
    pd_value = float(_scorecard.predict_proba(X)[0]) if hasattr(_scorecard, "feature_cols") \
        else float(_scorecard.predict_proba(req.loan.to_model_dict()))
    lgd = _lgd.predict(req.loan.to_model_dict())
    ead = req.loan.loan_amnt
    term_yrs = (36 if "36" in req.loan.term else 60) / 12.0
    el = expected_loss(pd_value, lgd, ead)
    irb = basel_rwa(pd_value, lgd, ead, maturity_years=term_yrs)
    credit_resp = ExpectedLossResponse(
        pd=pd_value, lgd=lgd, ead=ead, expected_loss=el,
        rwa=irb["RWA"], K=irb["K"],
        model_versions={"pd": _scorecard.version, "lgd": _lgd.version},
    )

    # Portfolio credit
    pc = simulate_portfolio_losses(pd=req.portfolio_pd, rho=req.portfolio_rho)
    pc_resp = PortfolioCreditResponse(**pc)

    return RiskReportResponse(
        market=market_resp,
        loan_expected_loss=credit_resp,
        portfolio_credit=pc_resp,
    )
