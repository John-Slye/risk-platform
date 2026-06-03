# Risk Platform

**Integrated Credit & Market Risk Analytics Platform**

A production-style risk system combining seven Value-at-Risk methodologies,
a full credit-risk stack (PD scorecard + XGBoost, LGD, Basel IRB Expected
Loss, Portfolio Credit VaR via Vasicek and Gaussian/t copulas), all behind
a FastAPI REST backend with a Streamlit dashboard, containerized via Docker
Compose, tested in GitHub Actions.

The promise: `docker compose up` and you can click through a real risk
platform in your browser in 90 seconds.

> Built end-to-end as a portfolio project: no pre-built VaR or credit-risk
> libraries, every model from scratch. Market risk engine ported in from
> the standalone [portfolio-var-framework](https://github.com/John-Slye/portfolio_var_project).

---

## Quick Start

Requires Docker Desktop. From the project root:

```bash
docker compose up --build
```

Then open in your browser:

- Dashboard:  http://localhost:8501
- API docs:   http://localhost:8000/docs (interactive Swagger UI)

Stop with `Ctrl-C` and `docker compose down`.

### Run locally without Docker

```bash
uv sync
uv run uvicorn risk_platform.api.main:app --reload &
uv run streamlit run src/risk_platform/dashboard/Home.py
```

### Train the credit models (optional, requires the LendingClub dataset)

```bash
# One-time data download via Kaggle API (~1.4 GB)
uv run kaggle datasets download -d wordsforthewise/lending-club -p data/raw --unzip
uv run python -m risk_platform.data.lending_club    # ~30 sec, cached
uv run python scripts/train_pd_scorecard.py         # ~60 sec
uv run python scripts/train_pd_xgboost.py           # ~60 sec
uv run python scripts/train_lgd_model.py            # ~30 sec
uv run python scripts/compare_pd_models.py          # Side-by-side report
```

After training, the FastAPI endpoints auto-load the pickles and serve real
predictions. Without trained pickles, the endpoints fall back to calibrated
stub responses (so the platform is always demoable, even on a clean clone).

---

## Architecture

```
+-------------------------------------------------------------+
|        STREAMLIT DASHBOARD  (port 8501)                     |
|        Home | Market Risk | Credit Risk | Portfolio Credit  |
|        Stress Testing | Model Cards | Portfolio Upload      |
+-------------------------------------------------------------+
                            | HTTP (requests)
                            v
+-------------------------------------------------------------+
|        FASTAPI BACKEND  (port 8000, OpenAPI at /docs)       |
|  /market/var  /market/es  /market/stress                    |
|  /credit/pd   /credit/lgd  /credit/expected_loss            |
|  /credit/portfolio_el       (bulk EL across a CSV)          |
|  /portfolio/credit_var  /risk_report                        |
|  /healthz   /version                                        |
+-------------------------------------------------------------+
        |                   |                    |
        v                   v                    v
+-----------+   +-------------------+   +------------------+
|  Market   |   |  Credit risk      |   | Portfolio credit |
|  Risk     |   |  PD (scorecard)   |   |  Vasicek ASRF    |
|  Project1 |   |  PD (XGBoost)     |   |  Gaussian copula |
|  engine,  |   |  LGD (XGB regr.)  |   |  Student-t copula|
|  ported   |   |  EL + Basel IRB   |   |  Credit VaR / EC |
+-----------+   +-------------------+   +------------------+
                            |
                            v
+-------------------------------------------------------------+
|  Data layer: pandas + parquet cache                         |
|  Market: yfinance, 9-asset multi-asset portfolio            |
|  Credit: LendingClub 2007-2018 (~2M loans)                  |
+-------------------------------------------------------------+

Tested with pytest (37 tests) | Linted with ruff | CI on GitHub Actions
```

---

## Headline Results

### Credit risk — LendingClub PD models (2014-2018 vintages, ~1.1M loans)

Time-based train (2014-16) / val (2017) / test (2018) splits. No look-ahead.

| Metric | Scorecard (WOE+LR) | XGBoost | Δ |
|---|---:|---:|---|
| Test AUC | 0.693 | **0.713** | +2.0 pp |
| Test KS  | 0.284 | **0.314** | +3.0 pp |
| Test Gini| 0.386 | **0.426** | +0.040 |

**Reading:** XGBoost adds ~2 percentage points of AUC over the WOE scorecard, which is the textbook outcome on tabular credit data. The improvement comes from feature interactions the linear scorecard cannot capture. The scorecard remains preferred for regulated decisions because adverse-action explanations are required to be feature-attributable, and a 600/PDO=20 scorecard is decomposable by construction.

**Top features by Information Value:** `int_rate` (0.51), `term` (0.21), `fico` (0.12), `dti` (0.08).

**Stability over time:** PSI (train -> test) = 0.030. Score distribution is stable. Calibration drift is more interesting: the model over-predicts default in the top decile by ~14 percentage points on the 2018 vintage because 2018 loans aren't fully matured (vintage drift). Acknowledged as a finding rather than a bug, with discussion of two-stage observation-window remediation in the methodology doc.

### Credit risk — portfolio Credit VaR

1,000-obligor sub-prime consumer book, PD = 20%, LGD = 92%, ρ = 0.10, $15M EAD:

| Method | Expected Loss | 99.9% Credit VaR | Economic Capital | RWA Density |
|---|---:|---:|---:|---:|
| Vasicek ASRF (analytical) | $2.76M | $7.68M | $4.92M | n/a |
| Gaussian copula (100k sims)| $2.76M | $7.81M | $5.05M | ~50% |
| **Student-t copula (df=5)** | $2.76M | **$9.20M** | **$6.44M** | n/a |

**Reading:** EL is identical across methods because expected loss is a function of marginals only (a hard invariant tested via `tests/test_portfolio_credit.py`). Vasicek and Gaussian MC agree to ~1.6% — proof that Basel IRB is the asymptotic limit of Vasicek-Gaussian. The Student-t copula adds **28% to Economic Capital** vs Gaussian under identical inputs — the central post-2008 modeling story (David X. Li paper / subprime tranches under-priced joint extreme defaults).

### Market risk — five-method VaR (2014-2026, 9-asset portfolio)

Out-of-sample backtests (rolling 500/1000-day window, refit GARCH every 60 days):

| Method | 99% Exc Rate | Kupiec p | Christof-ind p | Verdict |
|---|---:|---:|---:|---|
| Historical          | 1.30% | 0.142 | 0.001 | Right rate, clustered failures |
| Parametric (Normal) | **2.41%** | **0.000** | 0.000 | Over-exceeds by 2.4× |
| Monte Carlo (Normal)| 2.29% | 0.000 | 0.000 | Same |
| **FHS (GARCH(1,1))**| **1.13%** | **0.547** | 0.029 | **Only model that passes** |

**Reading:** The fat-tail problem materializes as a regulatory-grade failure for Normal-based methods. FHS is the only model that passes both Kupiec (right rate) and Christoffersen (non-clustered failures) at both 95% and 99%.

### Market risk — Extreme Value Theory

POT-GPD fit on the 5%-worst losses yields **ξ = 0.31** (heavy tail confirmed). Extrapolated 99.9% VaR = **4.14%**, more than **2× the Parametric-Normal estimate** of ~2.05% at the same confidence — quantifies the structural Gaussian underestimate at the deep tail.

---

## Methodology highlights

**PD scorecard.** `optbinning.BinningProcess` for WOE/IV with monotonic-trend enforcement. Logistic regression via `statsmodels.Logit` (preferred over scikit-learn because regulators want p-values and standard errors). Scaled to 600/PDO=20 scorecard using the industry-standard formula `score = offset - factor × ln(odds_of_default)`.

**XGBoost PD.** `XGBClassifier` with `enable_categorical=True`, early stopping on the validation set, frozen train-time categorical vocabulary to prevent unseen-category errors at inference. SHAP-ready (not run by default; the `feature_importance("gain")` method is the demo proxy).

**LGD model.** Single XGBoost regressor on `realized_lgd = 1 - recoveries/funded_amnt`. Predictions clipped to `[0.05, 0.95]`. The empirical LGD distribution on LendingClub is concentrated near 1.0 (mean 0.92, 37% at exactly 1.0, basically 0% at 0), which justifies a single-stage regressor over the two-stage model that would be needed when there is mass at both boundaries.

**Vasicek ASRF.** `conditional_pd(pd, ρ, α) = Φ((Φ⁻¹(pd) + √ρ · Φ⁻¹(α)) / √(1-ρ))`. Identical formula to Basel III IRB.

**Copula Monte Carlo.** One-factor structure: `A_i = √ρ · M + √(1-ρ) · ε_i`. Gaussian draws `M, ε_i` from `N(0,1)`. Student-t additionally divides by `√(χ²_df/df)` to inject joint-tail dependence. Default if `A_i < threshold` where threshold is `Φ⁻¹(PD)` (Gaussian) or `T_df⁻¹(PD)` (Student-t).

**Filtered Historical Simulation.** GARCH(1,1) on the portfolio return series. Standardize residuals to remove vol clustering. Resample empirically, rescale by current σ. Same GARCH refit cadence (every 60 trading days) used in backtesting.

**Backtesting.** Kupiec POF, Christoffersen independence, Christoffersen conditional coverage, Basel traffic-light. Strictly out-of-sample with refit at every change of regime.

**Stress testing.** Historical scenario re-pricing under 2008 GFC, 2020 COVID, 2018 Q4, 2022 rates, 2015 China devaluation windows. Extended price history (2007+) downloaded once for the GFC.

---

## Roadmap & status

| Phase | Status | Focus |
|---|---|---|
| 0 — Foundation             | shipped | Repo, packaging, FastAPI + Streamlit wired, Docker compose, CI |
| 1 — PD models              | shipped | WOE/IV scorecard + XGBoost, side-by-side validation report |
| 2 — LGD + Expected Loss    | shipped | LGD regressor, EL = PD·LGD·EAD, Basel IRB RWA, portfolio aggregator |
| 3 — Portfolio credit       | shipped | Vasicek ASRF + Gaussian/t copula MC + Credit VaR + Economic Capital |
| 4 — Market risk integration| shipped | Project 1 VaR/ES engine ported in as `src/risk_platform/market` |
| 5 — FastAPI deepening      | shipped | All endpoints real, Pydantic schemas, async-safe, OpenAPI at /docs |
| 6 — Dashboard polish       | shipped | Plotly charts, portfolio CSV upload, sample portfolio bundled |
| 7 — Production polish      | shipped | 37-test math suite, this README, methodology doc, demo video |

---

## Tech stack

| Layer | Tools |
|---|---|
| Language        | Python 3.12 |
| ML              | XGBoost, scikit-learn, optbinning (WOE/IV) |
| Stats           | scipy, statsmodels, arch (GARCH) |
| API             | FastAPI + Pydantic v2 + Uvicorn |
| Dashboard       | Streamlit + Plotly |
| Container       | Docker + Docker Compose |
| CI              | GitHub Actions |
| Test / Lint     | pytest, ruff |
| Package mgmt    | uv |

---

## Project structure

```
risk-platform/
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── pyproject.toml
├── .github/workflows/ci.yml
├── src/risk_platform/
│   ├── api/                    # FastAPI app
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── routers/{market,credit,portfolio,report}.py
│   ├── dashboard/              # Streamlit app
│   │   ├── Home.py
│   │   └── pages/              # 1_Market_Risk, 2_Credit_Risk, 3_Portfolio_Credit,
│   │                           # 4_Stress_Testing, 5_Model_Cards, 6_Portfolio_Upload
│   ├── credit/                 # PD scorecard + XGBoost, LGD, EL, Basel IRB
│   ├── portfolio/              # Vasicek ASRF, Gaussian/t copula MC, Credit VaR
│   ├── market/                 # ported from Project 1 (VaR, ES, FHS, EVT, stress)
│   ├── data/                   # LendingClub loader + processed parquet
│   └── core/                   # shared utilities
├── scripts/                    # train_pd_scorecard, train_pd_xgboost, train_lgd_model,
│                               # compare_pd_models, run_portfolio_credit_risk,
│                               # generate_sample_portfolio
├── tests/                      # 37 pytest tests across math + API layers
├── notebooks/                  # exploration + validation reports
├── data/sample/                # bundled 50-loan sample portfolio CSV
├── data/raw/                   # LendingClub CSV (gitignored)
├── data/processed/             # cached parquet (gitignored)
├── data/market/                # cached price parquet (gitignored)
├── models/                     # trained .pkl files (gitignored - retrain locally)
└── docs/                       # methodology + screenshots
```

---

## Honest limitations

(Disclosure section, in lieu of overclaiming.)

1. The platform's PD/LGD models load **trained pickles when present, stub responses otherwise.** A fresh clone running `docker compose up` will get calibrated stub PDs, not real predictions, until you run the training scripts and copy the pickles into the API image (or rebuild). This is documented behavior, not a bug — it keeps the platform demoable without forcing a 1.4 GB data download.
2. **LGD has near-zero R²** on the 2018 out-of-time test set. This is structural for unsecured consumer credit, where recoveries are dominated by post-default events (collections strategy, bankruptcy filings, secondary-market pricing) not observable at origination. The model captures the calibrated mean (0.92) but adds little discriminative signal.
3. **Vintage maturity bias** in the 2018 LendingClub vintage: loans that will eventually default are still `Current` in the snapshot, so the test base rate (15.8%) is artificially lower than train (20.8%). The right fix is a fixed observation window (e.g., default-within-24-months). Acknowledged in the methodology doc.
4. The dashboard `/portfolio/credit_var` page assumes a **homogeneous portfolio** (one PD, one LGD, one EAD applied uniformly). The Phase 6 portfolio-upload page is where you handle a heterogeneous book of real loans.

---

## License

MIT (LICENSE file to be added).
