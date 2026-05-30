"""Pydantic schemas: typed request/response models for every endpoint.

Why: FastAPI uses these to (1) parse and validate incoming JSON, (2) generate
auto-docs at /docs, (3) serialize Python objects back to JSON. If you ever
get a 422 error from the API, it's a Pydantic validation failure here.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------- Market risk -----------------------------------------------------
class MarketVaRRequest(BaseModel):
    method: Literal[
        "historical", "parametric_normal", "parametric_t",
        "monte_carlo_normal", "monte_carlo_t", "fhs",
    ] = "historical"
    alpha: float = Field(default=0.05, gt=0, lt=1,
                         description="Tail probability (0.05 = 95% VaR)")


class MarketVaRResponse(BaseModel):
    VaR: float
    method: str
    alpha: float
    model: str


class MarketStressRequest(BaseModel):
    scenario: Literal["2020_covid", "2008_gfc", "2022_rates"] = "2020_covid"


class MarketStressResponse(BaseModel):
    scenario: str
    cum_loss: float
    worst_day: float
    ann_vol: float
    model: str


# ---------- Credit risk -----------------------------------------------------
class LoanFeatures(BaseModel):
    """Single loan's features for PD/LGD inference."""
    annual_income: float = Field(gt=0, description="USD")
    loan_amount: float = Field(gt=0, description="USD")
    interest_rate: float = Field(gt=0, lt=1, description="Decimal, e.g. 0.10")
    term_months: int = Field(gt=0)
    fico: int = Field(ge=300, le=850)
    dti: float = Field(ge=0, lt=1, description="Debt-to-income ratio (decimal)")
    purpose: str = Field(default="debt_consolidation")
    home_ownership: str = Field(default="RENT")


class PDRequest(BaseModel):
    loan: LoanFeatures
    model: Literal["scorecard", "xgboost"] = "scorecard"


class PDResponse(BaseModel):
    pd: float
    score: Optional[int] = None
    model: str


class LGDResponse(BaseModel):
    lgd: float
    model: str


class ExpectedLossResponse(BaseModel):
    pd: float
    lgd: float
    ead: float
    expected_loss: float
    rwa: float
    K: float = Field(description="Basel capital requirement per $1 of EAD")
    model_versions: dict


# ---------- Portfolio credit ------------------------------------------------
class PortfolioCreditRequest(BaseModel):
    pd: float = Field(default=0.05, gt=0, lt=1)
    rho: float = Field(default=0.15, gt=0, lt=1,
                       description="Asset correlation")
    n_obligors: int = Field(default=1000, gt=0)
    lgd: float = Field(default=0.45, ge=0, le=1)
    copula: Literal["gaussian", "t"] = "gaussian"
    df: int = Field(default=5, gt=2, description="t-copula degrees of freedom")
    n_simulations: int = Field(default=10_000, gt=0, le=1_000_000)


class PortfolioCreditResponse(BaseModel):
    copula: str
    df: Optional[int] = None
    expected_loss: float
    credit_var_99: float
    credit_var_99_9: float
    economic_capital: float
    n_simulations: int
    model: str


# ---------- Combined risk report -------------------------------------------
class RiskReportRequest(BaseModel):
    market_method: str = "historical"
    market_alpha: float = 0.05
    loan: LoanFeatures
    portfolio_pd: float = 0.05
    portfolio_rho: float = 0.15


class RiskReportResponse(BaseModel):
    market: MarketVaRResponse
    loan_expected_loss: ExpectedLossResponse
    portfolio_credit: PortfolioCreditResponse


# ---------- Platform meta ---------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"


class VersionResponse(BaseModel):
    platform_version: str
    models: dict[str, str]
