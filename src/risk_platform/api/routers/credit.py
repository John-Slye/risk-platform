"""Credit risk endpoints: /credit/pd, /credit/lgd, /credit/expected_loss."""

from __future__ import annotations

from fastapi import APIRouter, Query

from risk_platform.api.schemas import (
    ExpectedLossResponse, LGDResponse, LoanFeatures, PDRequest, PDResponse,
)
from risk_platform.credit import (
    LGDModel, ScorecardPD, XGBoostPD, basel_rwa, expected_loss,
)


router = APIRouter(prefix="/credit", tags=["credit"])


@router.post("/pd", response_model=PDResponse)
def post_pd(req: PDRequest) -> PDResponse:
    """Predict probability of default for a single loan."""
    m = ScorecardPD() if req.model == "scorecard" else XGBoostPD()
    features = req.loan.model_dump()
    pd = m.predict_proba(features)
    score = m.score(features) if isinstance(m, ScorecardPD) else None
    return PDResponse(pd=pd, score=score, model=m.version)


@router.post("/lgd", response_model=LGDResponse)
def post_lgd(loan: LoanFeatures) -> LGDResponse:
    """Predict loss given default."""
    m = LGDModel()
    return LGDResponse(lgd=m.predict(loan.model_dump()), model=m.version)


@router.post("/expected_loss", response_model=ExpectedLossResponse)
def post_expected_loss(
    loan: LoanFeatures,
    pd_model: str = Query("scorecard", pattern="^(scorecard|xgboost)$"),
) -> ExpectedLossResponse:
    """End-to-end loan loss: PD * LGD * EAD plus Basel IRB capital."""
    pd_eng = ScorecardPD() if pd_model == "scorecard" else XGBoostPD()
    lgd_eng = LGDModel()
    features = loan.model_dump()
    pd = pd_eng.predict_proba(features)
    lgd = lgd_eng.predict(features)
    ead = loan.loan_amount
    el = expected_loss(pd, lgd, ead)
    irb = basel_rwa(pd, lgd, ead, maturity_years=loan.term_months / 12.0)
    return ExpectedLossResponse(
        pd=pd, lgd=lgd, ead=ead, expected_loss=el,
        rwa=irb["RWA"], K=irb["K"],
        model_versions={"pd": pd_eng.version, "lgd": lgd_eng.version},
    )
