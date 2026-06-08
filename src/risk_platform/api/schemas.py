"""Pydantic schemas: typed request/response models for every endpoint.

Why: FastAPI uses these to (1) parse and validate incoming JSON, (2) generate
auto-docs at /docs, (3) serialize Python objects back to JSON. If you ever
get a 422 error from the API, it's a Pydantic validation failure here.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------- Market risk -----------------------------------------------------
class MarketPortfolio(BaseModel):
    """Custom market portfolio: per-asset ticker + weight."""
    tickers: list[str] = Field(min_length=2, max_length=30,
        description="Yahoo Finance tickers (e.g. ['SPY', 'AAPL', 'TLT'])")
    weights: list[float] = Field(min_length=2, max_length=30,
        description="Portfolio weights; will be normalized to sum to 1")


class MarketVaRRequest(BaseModel):
    method: Literal[
        "historical", "parametric_normal", "parametric_t",
        "monte_carlo_normal", "monte_carlo_t", "fhs",
    ] = "historical"
    alpha: float = Field(default=0.05, gt=0, lt=1,
                         description="Tail probability (0.05 = 95% VaR)")
    portfolio: Optional[MarketPortfolio] = Field(default=None,
        description="Optional custom portfolio; default 9-asset book is used if omitted")


class MarketVaRResponse(BaseModel):
    VaR: float
    method: str
    alpha: float
    model: str


class MarketStressRequest(BaseModel):
    scenario: Literal["2020_covid", "2008_gfc", "2022_rates"] = "2020_covid"
    portfolio: Optional[MarketPortfolio] = Field(default=None,
        description="Optional custom portfolio; default 9-asset book is used if omitted")


class MarketStressResponse(BaseModel):
    scenario: str
    cum_loss: float
    worst_day: float
    ann_vol: float
    model: str


# ---------- Credit risk -----------------------------------------------------
class LoanFeatures(BaseModel):
    """Single loan's features for PD/LGD inference.

    Field names mirror the LendingClub schema (loan_amnt, annual_inc, etc.)
    so the API contract matches the training data exactly. Required fields
    are the strongest predictors; the rest carry sensible defaults that
    represent a 'typical' loan, so a demo caller can ignore them.
    """
    # Required (highest-IV features)
    loan_amnt: float = Field(gt=0, description="Principal in USD")
    int_rate: float = Field(gt=0, lt=100, description="Annual percent, e.g. 10.5")
    term: str = Field(default="36 months",
                      description="'36 months' or '60 months'")
    annual_inc: float = Field(gt=0, description="Annual income USD")
    fico: int = Field(ge=300, le=850, description="FICO midpoint at origination")
    dti: float = Field(ge=0, lt=100, description="Debt-to-income, percent")

    # Optional with defaults (lower IV; defaults represent a typical loan)
    installment: Optional[float] = Field(default=None,
        description="Monthly payment USD; computed from amnt/rate/term if not given")
    delinq_2yrs: int = Field(default=0)
    inq_last_6mths: int = Field(default=0)
    open_acc: int = Field(default=8)
    pub_rec: int = Field(default=0)
    revol_bal: float = Field(default=10_000)
    revol_util: float = Field(default=40.0, description="Revolving utilization %")
    total_acc: int = Field(default=20)
    mort_acc: int = Field(default=0)
    pub_rec_bankruptcies: int = Field(default=0)
    emp_length: str = Field(default="5 years")
    home_ownership: str = Field(default="RENT")
    verification_status: str = Field(default="Verified")
    purpose: str = Field(default="debt_consolidation")
    application_type: str = Field(default="Individual")

    def to_model_dict(self) -> dict:
        """Pack into a dict matching exactly what ScorecardPD/XGBoostPD expect."""
        d = self.model_dump()
        # Compute installment if missing: standard amortization formula.
        if d.get("installment") is None:
            r = self.int_rate / 100.0 / 12.0
            n = 36 if "36" in self.term else 60
            d["installment"] = (
                self.loan_amnt * r / (1 - (1 + r) ** (-n)) if r > 0
                else self.loan_amnt / n
            )
        return d


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
class Obligor(BaseModel):
    """A single obligor in a heterogeneous credit portfolio."""
    pd: float = Field(gt=0, lt=1)
    lgd: float = Field(ge=0, le=1)
    ead: float = Field(gt=0)


class PortfolioCreditRequest(BaseModel):
    # Homogeneous portfolio (used if `obligors` is None)
    pd: float = Field(default=0.20, gt=0, lt=1)
    rho: float = Field(default=0.10, gt=0, lt=1,
                       description="Asset correlation (Basel retail ~0.04, sub-prime ~0.10-0.15)")
    n_obligors: int = Field(default=1000, gt=0)
    lgd: float = Field(default=0.92, ge=0, le=1,
                       description="Per-obligor LGD (unsecured consumer ~0.90)")
    ead: float = Field(default=15_000.0, gt=0,
                       description="Per-obligor exposure in USD")
    copula: Literal["gaussian", "t"] = "gaussian"
    df: int = Field(default=5, gt=2, description="t-copula degrees of freedom")
    n_simulations: int = Field(default=10_000, gt=0, le=1_000_000)

    # Heterogeneous portfolio (per-obligor PD/LGD/EAD). When set, overrides
    # the homogeneous fields above.
    obligors: Optional[list[Obligor]] = Field(default=None,
        description="Optional per-obligor list; if provided, the homogeneous "
                    "pd/lgd/ead/n_obligors fields are ignored.")


class PortfolioCreditResponse(BaseModel):
    copula: str
    df: Optional[int] = None
    expected_loss: float
    credit_var_99: float
    credit_var_99_9: float
    economic_capital: float
    tail_es_99_9: Optional[float] = None
    n_simulations: int
    total_ead: Optional[float] = None
    hhi: Optional[float] = None
    histogram_bins: Optional[list[float]] = Field(
        default=None, description="Bin edges of the simulated loss distribution"
    )
    histogram_counts: Optional[list[int]] = Field(
        default=None, description="Count of simulations per histogram bin"
    )
    model: str


# Bulk portfolio EL endpoint -------------------------------------------------
class PortfolioELRequest(BaseModel):
    loans: list[LoanFeatures] = Field(
        description="List of loans (each with all required + optional fields)"
    )
    pd_model: Literal["scorecard", "xgboost"] = "scorecard"


class PortfolioELResponse(BaseModel):
    n_loans: int
    total_ead: float
    total_el: float
    total_rwa: float
    weighted_pd: float
    weighted_lgd: float
    el_pct_of_ead: float
    rwa_density: float
    per_loan_pds: list[float]
    per_loan_lgds: list[float]
    per_loan_eads: list[float]
    per_loan_els: list[float]
    per_loan_rwas: list[float]
    model_versions: dict


# ---------- Combined risk report -------------------------------------------
class RiskReportRequest(BaseModel):
    market_method: str = "historical"
    market_alpha: float = 0.05
    market_portfolio: Optional[MarketPortfolio] = Field(default=None,
        description="Optional custom market portfolio; default 9-asset book used if omitted")
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
