"""Credit risk endpoints: /credit/pd, /credit/lgd, /credit/expected_loss.

Loads the trained scorecard + XGBoost pickles at import time if they exist.
Falls back to the Phase 0 stubs otherwise (so CI without the pickles still
passes; the dashboard 'Backend status' surfaces which models are live).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Query

from risk_platform.api.schemas import (
    ExpectedLossResponse, LGDResponse, LoanFeatures, PDRequest, PDResponse,
)
from risk_platform.credit import LGDModel, basel_rwa, expected_loss
# Stubs (Phase 0)
from risk_platform.credit.pd_models import ScorecardPD as StubScorecard
from risk_platform.credit.pd_models import XGBoostPD as StubXGBoost
# Real, trained models (Phase 1)
from risk_platform.credit.pd_scorecard import ScorecardPD as RealScorecard
from risk_platform.credit.pd_xgboost import XGBoostPD as RealXGBoost


# Project-level models dir: ../../../../models from this file.
_MODELS_DIR = Path(__file__).resolve().parents[4] / "models"
_SC_PATH = _MODELS_DIR / "pd_scorecard.pkl"
_XG_PATH = _MODELS_DIR / "pd_xgboost.pkl"


def _load_scorecard():
    if _SC_PATH.exists():
        return RealScorecard.load(_SC_PATH)
    return StubScorecard()


def _load_xgboost():
    if _XG_PATH.exists():
        return RealXGBoost.load(_XG_PATH)
    return StubXGBoost()


# Singletons (loaded once at process start).
_scorecard = _load_scorecard()
_xgboost = _load_xgboost()
_lgd = LGDModel()


router = APIRouter(prefix="/credit", tags=["credit"])


def _loan_to_df(loan: LoanFeatures) -> pd.DataFrame:
    """Pack loan into a one-row DataFrame matching the model feature columns."""
    return pd.DataFrame([loan.to_model_dict()])


@router.post("/pd", response_model=PDResponse)
def post_pd(req: PDRequest) -> PDResponse:
    """Predict probability of default for a single loan."""
    m = _scorecard if req.model == "scorecard" else _xgboost
    X = _loan_to_df(req.loan)
    pd_value = float(m.predict_proba(X)[0]) if hasattr(m, "feature_cols") else \
        float(m.predict_proba(req.loan.to_model_dict()))
    score = None
    try:
        score = int(m.score(X)[0]) if hasattr(m, "feature_cols") else int(m.score(req.loan.to_model_dict()))
    except Exception:
        pass
    return PDResponse(pd=pd_value, score=score, model=m.version)


@router.post("/lgd", response_model=LGDResponse)
def post_lgd(loan: LoanFeatures) -> LGDResponse:
    """Predict loss given default."""
    return LGDResponse(lgd=_lgd.predict(loan.to_model_dict()), model=_lgd.version)


@router.post("/expected_loss", response_model=ExpectedLossResponse)
def post_expected_loss(
    loan: LoanFeatures,
    pd_model: str = Query("scorecard", pattern="^(scorecard|xgboost)$"),
) -> ExpectedLossResponse:
    """End-to-end loan loss: PD * LGD * EAD plus Basel IRB capital."""
    pd_eng = _scorecard if pd_model == "scorecard" else _xgboost
    X = _loan_to_df(loan)
    pd_value = float(pd_eng.predict_proba(X)[0]) if hasattr(pd_eng, "feature_cols") \
        else float(pd_eng.predict_proba(loan.to_model_dict()))
    lgd = _lgd.predict(loan.to_model_dict())
    ead = loan.loan_amnt
    term_yrs = (36 if "36" in loan.term else 60) / 12.0
    el = expected_loss(pd_value, lgd, ead)
    irb = basel_rwa(pd_value, lgd, ead, maturity_years=term_yrs)
    return ExpectedLossResponse(
        pd=pd_value, lgd=lgd, ead=ead, expected_loss=el,
        rwa=irb["RWA"], K=irb["K"],
        model_versions={"pd": pd_eng.version, "lgd": _lgd.version},
    )
