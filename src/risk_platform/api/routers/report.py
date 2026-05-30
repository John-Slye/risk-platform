"""Unified market + credit risk report: /risk_report."""

from __future__ import annotations

from fastapi import APIRouter

from risk_platform.api.schemas import (
    ExpectedLossResponse, MarketVaRResponse, PortfolioCreditResponse,
    RiskReportRequest, RiskReportResponse,
)
from risk_platform.credit import (
    LGDModel, ScorecardPD, basel_rwa, expected_loss,
)
from risk_platform.market import MarketRisk
from risk_platform.portfolio import simulate_portfolio_losses


router = APIRouter(prefix="", tags=["report"])


@router.post("/risk_report", response_model=RiskReportResponse)
def post_risk_report(req: RiskReportRequest) -> RiskReportResponse:
    """Run market VaR + loan EL + portfolio Credit VaR and return all three."""
    # Market
    mr_engine = MarketRisk()
    mr = mr_engine.var(req.market_method, req.market_alpha)
    market_resp = MarketVaRResponse(**mr)

    # Credit (loan-level)
    pd_eng = ScorecardPD()
    lgd_eng = LGDModel()
    features = req.loan.model_dump()
    pd = pd_eng.predict_proba(features)
    lgd = lgd_eng.predict(features)
    ead = req.loan.loan_amount
    el = expected_loss(pd, lgd, ead)
    irb = basel_rwa(pd, lgd, ead, maturity_years=req.loan.term_months / 12.0)
    credit_resp = ExpectedLossResponse(
        pd=pd, lgd=lgd, ead=ead, expected_loss=el,
        rwa=irb["RWA"], K=irb["K"],
        model_versions={"pd": pd_eng.version, "lgd": lgd_eng.version},
    )

    # Portfolio credit
    pc = simulate_portfolio_losses(pd=req.portfolio_pd, rho=req.portfolio_rho)
    pc_resp = PortfolioCreditResponse(**pc)

    return RiskReportResponse(
        market=market_resp,
        loan_expected_loss=credit_resp,
        portfolio_credit=pc_resp,
    )
