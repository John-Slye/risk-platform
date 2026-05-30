# Risk Platform

Integrated Credit & Market Risk Analytics Platform.

A production-style risk system combining market VaR/ES (Historical, Parametric,
Monte Carlo, GARCH-filtered Historical, EVT), credit PD/LGD/Expected Loss
modeling, portfolio Credit VaR via Vasicek + copulas, and Basel IRB capital,
all behind a FastAPI REST API with a Streamlit dashboard, containerized with
Docker, tested in GitHub Actions.

> **Status: Phase 0 (walking skeleton) shipped.** All endpoints return stubs;
> phases 1 through 7 replace the stubs with real models. Roadmap below.

## Quick start

Requires Docker Desktop. From the project root:

```bash
docker compose up --build
```

Then open:

- API docs (Swagger UI):  http://localhost:8000/docs
- Dashboard:              http://localhost:8501

Stop with `Ctrl-C` and `docker compose down`.

### Run locally without Docker

```bash
uv sync
uv run uvicorn risk_platform.api.main:app --reload &
uv run streamlit run src/risk_platform/dashboard/Home.py
```

### Run the test suite

```bash
uv run pytest -q
```

## Architecture

```
+--------------------------------------------------------+
|       STREAMLIT DASHBOARD  (port 8501, Phase 6)        |
|       Home  ·  Market  ·  Credit  ·  Portfolio  ·       |
|       Stress  ·  Model Cards                            |
+--------------------------------------------------------+
                          | HTTP (requests)
                          v
+--------------------------------------------------------+
|        FASTAPI BACKEND  (port 8000, Phase 5)           |
|  /market/var  /market/es  /market/stress               |
|  /credit/pd   /credit/lgd  /credit/expected_loss       |
|  /portfolio/credit_var  /risk_report                   |
|  /healthz  /version  /docs (Swagger UI)                |
+--------------------------------------------------------+
        |               |                 |
        v               v                 v
+----------+   +------------------+   +-----------------+
|  Market  |   |  Credit (loan)   |   | Portfolio Credit|
|  Risk    |   |  PD (LR)         |   |  Vasicek        |
|  Phase 4 |   |  PD (XGBoost)    |   |  Copulas (G + t)|
|  ported  |   |  LGD             |   |  Credit VaR     |
|  from P1 |   |  EL + Basel RWA  |   |  Econ Capital   |
+----------+   +------------------+   +-----------------+
                          |
                          v
+--------------------------------------------------------+
|  Data layer  ·  pandas  ·  SQLite (future)  ·          |
|  yfinance (market)  ·  LendingClub CSV (credit)        |
+--------------------------------------------------------+

Wrapped with: Docker + Docker Compose · GitHub Actions CI · pytest · ruff
```

## Roadmap

| Phase | Status | Focus |
|---|---|---|
| 0 — Foundation        | shipped | Repo, package skeleton, stub models, FastAPI + Streamlit wired, Docker compose, CI |
| 1 — PD models         | pending | Logistic-regression scorecard with WOE/IV + XGBoost with optuna + SHAP on LendingClub |
| 2 — LGD + Expected Loss | pending | LGD via beta regression or two-stage; EL = PD·LGD·EAD; Basel IRB RWA |
| 3 — Portfolio credit  | pending | Vasicek ASRF + Gaussian/t-copula Monte Carlo; Credit VaR; Economic Capital |
| 4 — Market risk       | pending | Port Project 1's VaR/ES framework as `src/risk_platform/market/` |
| 5 — FastAPI deepening | pending | All endpoints real, request validation, async where useful |
| 6 — Dashboard polish  | pending | Plotly charts, sample portfolio CSV, PDF report export |
| 7 — Production polish | pending | ~70% test coverage on math, integration tests, demo video, model methodology doc |

## Tech stack

| Layer | Tools |
|---|---|
| Language        | Python 3.12 |
| ML              | XGBoost, scikit-learn, optbinning (WOE/IV) |
| Stats           | scipy, statsmodels, arch (GARCH) |
| API             | FastAPI + Pydantic v2 + Uvicorn |
| Dashboard       | Streamlit + Plotly |
| DB              | SQLite + SQLAlchemy (future phases) |
| Container       | Docker + Docker Compose |
| CI              | GitHub Actions |
| Test / Lint     | pytest, ruff, black |
| Package mgmt    | uv |

## Project structure

```
risk-platform/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── pyproject.toml             # uv-managed
├── .github/workflows/ci.yml   # GitHub Actions
├── src/risk_platform/
│   ├── api/                   # FastAPI app
│   │   ├── main.py
│   │   ├── schemas.py         # Pydantic models
│   │   └── routers/{market,credit,portfolio,report}.py
│   ├── dashboard/             # Streamlit app
│   │   ├── Home.py
│   │   └── pages/             # 1_Market_Risk, 2_Credit_Risk, ...
│   ├── credit/                # PD, LGD, EL, Basel
│   ├── portfolio/             # Vasicek, copulas
│   ├── market/                # ported from Project 1 in Phase 4
│   ├── data/                  # ingestion, schemas
│   └── core/                  # shared utilities
├── tests/                     # pytest
├── notebooks/                 # exploration + validation reports
├── data/                      # gitignored
└── docs/                      # methodology, model cards
```

## What Phase 0 proves

End-to-end pipeline works: `docker compose up` builds and runs both services,
the dashboard reaches the API, every endpoint returns a typed Pydantic
response, CI runs the test suite on every push. Each later phase swaps a
stub for real math without touching the surrounding plumbing.

## License

MIT (see LICENSE, to be added in Phase 7 polish).
