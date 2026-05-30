"""FastAPI application entry point.

Run locally:   uv run uvicorn risk_platform.api.main:app --reload
Open browser:  http://localhost:8000/docs   (interactive Swagger UI)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from risk_platform import __version__
from risk_platform.api.routers import credit, market, portfolio, report
from risk_platform.api.schemas import HealthResponse, VersionResponse
from risk_platform.credit import ScorecardPD, XGBoostPD, LGDModel


app = FastAPI(
    title="Integrated Credit & Market Risk Analytics Platform",
    description=(
        "REST API for market VaR/ES, credit PD/LGD/EL, and portfolio Credit VaR. "
        "Auto-generated docs available at /docs and /redoc."
    ),
    version=__version__,
)

# CORS so the Streamlit dashboard (different port) can talk to us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the per-domain routers.
app.include_router(market.router)
app.include_router(credit.router)
app.include_router(portfolio.router)
app.include_router(report.router)


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz() -> HealthResponse:
    """Liveness probe — returns 200 OK if the API process is responsive."""
    return HealthResponse(status="ok")


@app.get("/version", response_model=VersionResponse, tags=["meta"])
def version() -> VersionResponse:
    """Platform + model version metadata."""
    return VersionResponse(
        platform_version=__version__,
        models={
            "scorecard_pd": ScorecardPD().version,
            "xgboost_pd":   XGBoostPD().version,
            "lgd":          LGDModel().version,
        },
    )
