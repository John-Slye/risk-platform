# Interview Prep: Integrated Credit & Market Risk Analytics Platform

A study sheet for credit-risk, quant-developer, and risk-technology interviews. Mirrors the Project 1 prep doc but is credit-focused. Read the platform's README first; this assumes you remember the numbers.

## How to use this doc

Practice every answer **out loud**, under 60 seconds, without reading. The numbers from your runs are baked into each answer so you can quote them confidently. Structure is always: define the concept, state the result you got, explain why it matters.

---

## 1. The 90-second elevator pitch

> "I built a production-style risk analytics platform that integrates credit risk and market risk in one system. The credit stack has two PD models (a logistic-regression scorecard with WOE/IV feature engineering, and an XGBoost benchmark), an LGD regressor, Basel IRB Expected Loss, and portfolio Credit VaR via Vasicek ASRF and Gaussian/t-copula Monte Carlo. The market stack is the standalone VaR project I built first, refactored as a platform module. Everything sits behind a FastAPI backend with a Streamlit dashboard, all containerized with Docker Compose. The headline credit finding is that the Student-t copula adds 28% to Economic Capital versus Gaussian on a sub-prime consumer book — the central post-2008 modeling story. The headline market finding is that Filtered Historical Simulation with GARCH(1,1) is the only method out of five that passes both Kupiec and Christoffersen backtests at 95% and 99%. The platform itself is `docker compose up` away from a clickable demo."

Memorize the structure: scope, two stacks, headline credit finding, headline market finding, demo-able.

---

## 2. Core definitions to have on instant recall

**PD (Probability of Default).** Likelihood a borrower fails to meet payment obligations over a defined horizon (typically 12 months). Modeled at origination using borrower features (FICO, DTI, loan amount, term, employment, etc.).

**LGD (Loss Given Default).** Fraction of exposure lost when default occurs. Bounded in [0, 1]. For unsecured consumer credit, often near 1. For secured commercial real estate, 0.3-0.5.

**EAD (Exposure at Default).** Dollar amount at risk if the borrower defaults. For term loans, ≈ outstanding balance. For credit lines, EAD = drawn + CCF × undrawn, where CCF (Credit Conversion Factor) estimates how much of the unused line gets drawn before default.

**Expected Loss.** EL = PD × LGD × EAD. The average loss per loan over the horizon. Banks book this through the income statement as provisions.

**Unexpected Loss.** Loss in excess of EL. Banks hold equity capital against this.

**Weight of Evidence (WOE).** Per-bin: `ln((goods% in bin) / (bads% in bin))`. Higher WOE = lower default risk. Replaces raw feature values to make the relationship with default rate monotonic and signed-log-odds-interpretable, which lets logistic regression capture non-linearities.

**Information Value (IV).** Summary of a feature's discriminatory power: `Σ over bins of (goods% - bads%) × WOE`. Cutoffs: <0.02 useless, 0.02-0.10 weak, 0.10-0.30 medium (most useful features), >0.30 strong but suspect (could be leakage).

**PDO (Points to Double the Odds).** Scoring convention: a score increment of PDO doubles the good:bad odds. Industry standard is PDO=20 with base score 600 at base odds 50:1.

**Population Stability Index (PSI).** Measures shift in score distribution between training and production. PSI = Σ (actual% - expected%) × ln(actual% / expected%). Cutoffs: < 0.10 stable, 0.10-0.25 monitor, > 0.25 retrain.

**Vasicek single-factor (ASRF) model.** Each obligor's latent asset value `A_i = √ρ · M + √(1-ρ) · ε_i`. M = systematic factor (the economy), ε_i = idiosyncratic shock, both N(0,1). Default if `A_i < Φ⁻¹(PD)`. The conditional PD given an α-quantile factor draw is `Φ((Φ⁻¹(PD) + √ρ · Φ⁻¹(α)) / √(1-ρ))`. This is exactly the Basel III IRB formula.

**Asset correlation ρ.** How much obligors' assets move together. Basel sets ρ = 0.12-0.24 for corporate, ~0.04 for retail. Sub-prime consumer typically 0.10-0.15.

**Copula.** A function that joins marginal distributions into a joint distribution. The Gaussian copula assumes joint extreme co-movements are exponentially rare; the Student-t copula has fatter joint tails (when one obligor defaults badly, others are more likely to default too).

**Credit VaR.** Loss exceeded with probability α on a portfolio loss distribution. Basel uses 99.9% for economic capital.

**Economic Capital.** EC = Credit VaR − Expected Loss. The unexpected-loss reserve a bank holds equity against.

**Basel IRB.** Internal Ratings-Based approach: banks supply their own PD (and possibly LGD, EAD) estimates, and the regulator's formula computes risk-weighted assets. Capital requirement = 8% × RWA.

---

## 3. The twelve standard interview questions

### Q1. Walk me through how you'd build a PD model from scratch.

1. **Target definition.** Define "default" precisely — typically a fixed observation window (e.g., charged-off-within-24-months-from-origination). Discard loans whose outcome is unknown.
2. **Train/validate/test splits, by time not random.** Older vintages train, recent vintages test. Random splits leak information across time because credit standards and economic conditions change.
3. **Feature selection.** Compute IV per candidate feature, keep IV in 0.02-0.30 range, drop > 0.5 IV as suspected leakage.
4. **WOE binning** with monotonicity enforced. Replace raw features with WOE values.
5. **Fit logistic regression** on WOE features. Use statsmodels for p-values/standard errors (regulators expect them).
6. **Calibration check.** AUC and KS for ranking. Calibration curves and Brier for absolute PD accuracy. Platt scaling or isotonic regression if calibration is off.
7. **PSI monitoring** on the score distribution out-of-time. PSI > 0.25 triggers retrain.
8. **Document everything.** Adverse-action explanations, model methodology, validation results. SR 11-7 requires it.

### Q2. What's the difference between PD, LGD, and EAD?

PD is the probability of default. LGD is the fraction of exposure lost given that default has occurred. EAD is the dollar amount at risk when default occurs. Expected Loss = PD × LGD × EAD.

In practice the three are estimated by different models on different data:
- PD: classification on origination features
- LGD: regression on recovery data
- EAD: typically equals the outstanding balance for term loans, modeled via CCF (Credit Conversion Factor) for credit lines

### Q3. What's Weight of Evidence and Information Value, and why do scorecards use them?

WOE per bin is `ln((goods% in bin) / (bads% in bin))`. It transforms a raw feature into a signed log-odds quantity that's monotonically related to default risk. Logistic regression is linear, so non-monotonic relationships between raw features and default are missed — WOE binning fixes that.

IV per feature is `Σ (goods% - bads%) × WOE`. It's a single-number summary of discriminative power, used for feature selection. Cutoffs: < 0.02 useless, 0.02-0.10 weak, 0.10-0.30 medium (sweet spot), > 0.30 strong (or suspect).

In my project the top features by IV were `int_rate` (0.51), `term` (0.21), `fico` (0.12), `dti` (0.08).

### Q4. Why time-based train/test splits in credit modeling?

Because economic conditions, credit standards, and underwriting policies change over time. A random K-fold split lets the model learn from days that occur both before and after the test points, which is impossible in production. Time-based splits preserve the actual deployment scenario.

In my project I trained on 2014-16 vintages, validated on 2017, tested on 2018. The 2018 vintage shows materially different statistics from 2017 (vintage maturity bias and macro shifts), which is exactly the kind of stress a deployment model would face.

### Q5. Why might regulators prefer a logistic scorecard to XGBoost even if XGBoost has higher AUC?

Four reasons.

1. **Regulatory acceptance.** SR 11-7 effectively requires model explainability for credit decisions. Scorecards are decomposable by feature into point contributions; gradient boosting needs SHAP or equivalent.
2. **Adverse action notices.** US fair-lending law requires lenders to explain declines. A scorecard naturally produces "you lost 15 points for high DTI"; XGBoost requires an extra interpretation layer.
3. **Stability.** Logistic on WOE features is highly stable. Tree ensembles can shift decision boundaries dramatically with small data changes, complicating PSI monitoring.
4. **Deployability.** A scorecard ultimately is a SQL `CASE WHEN` statement. Mainframe-friendly.

In practice, sophisticated lenders use both: XGBoost as a challenger for portfolio-level risk-based pricing, scorecard for the final auditable decision. In my project, XGBoost added only 2 AUC points over the scorecard — real money at portfolio scale but not enough to outweigh the regulatory cost for the unitary decision.

### Q6. What is the Population Stability Index and how do you interpret it?

PSI compares two score distributions bin by bin: `PSI = Σ (actual% - expected%) × ln(actual% / expected%)`. It quantifies how much the production score distribution has drifted from the training set.

Cutoffs: PSI < 0.10 means stable, no action. 0.10 - 0.25 means monitor closely. > 0.25 triggers a retrain.

In my project PSI from 2014-16 training to 2018 test was 0.030 — well under the monitor threshold. But that doesn't mean the model is fine, only that the score distribution shape is stable. The calibration drifted (top-decile predicted PD overshot realized by 14 percentage points) because the test base rate was lower than train. PSI catches distribution shape; calibration tables catch absolute level. They are different and both matter.

### Q7. Walk me through how a copula works in portfolio credit risk.

A copula joins marginal distributions into a joint distribution. The one-factor structure most commonly used in credit: every obligor's latent "asset value" `A_i = √ρ · M + √(1-ρ) · ε_i`, where M is the shared systematic factor and ε_i is the obligor-specific shock.

For the Gaussian copula, M and ε_i are N(0, 1). For the Student-t copula, you additionally divide the Gaussian latent by `√(χ²_df / df)` — same random scaling across all obligors within a simulation, which preserves tail dependence.

Default occurs when A_i falls below a threshold corresponding to the marginal PD: Φ⁻¹(PD) for Gaussian, T_df⁻¹(PD) for t-copula.

You then simulate N portfolios, count losses per simulation, and take quantiles of the resulting distribution to get Credit VaR and Economic Capital.

### Q8. What did the 2008 crisis reveal about the Gaussian copula?

David X. Li's 2000 paper popularized using Gaussian copulas to price CDO tranches. The Gaussian copula assumes joint extreme defaults are exponentially rare — when correlations are calibrated to historical data, the probability of many obligors defaulting together is tiny.

2008 revealed that this assumption was structurally wrong: in genuine stress (the housing crash), default correlations went toward 1. Mortgages defaulted en masse. AAA-rated CDO tranches that the Gaussian copula said should never lose money lost essentially everything. Banks had under-priced and under-capitalized these positions because their models said the risk wasn't there.

In my project I quantified this: on a sub-prime consumer book, Student-t copula (df=5) Economic Capital is 28% higher than Gaussian under identical inputs. That 28% is roughly the model-risk severity that the post-2008 regulatory framework was built to address.

### Q9. What is the Basel IRB formula? Why those inputs?

Basel III Internal Ratings-Based formula for capital charge per dollar of EAD:

```
K = LGD × [ Φ((Φ⁻¹(PD) + √ρ × Φ⁻¹(0.999)) / √(1-ρ)) - PD ] × M_adj
RWA = K × 12.5 × EAD
```

Inputs:
- **PD** — the bank's own estimate
- **LGD** — bank's or supervisor's
- **EAD** — outstanding plus CCF × undrawn
- **ρ** (asset correlation) — supervisor's formula based on PD and asset class
- **M_adj** (maturity adjustment) — scales capital up for longer-term exposures

The 0.999 is the supervisory worst-case factor draw (1-in-1000-year stress). The formula computes conditional PD under that stress and subtracts the expected PD (since EL is already provisioned through earnings, capital is for unexpected loss only).

This is exactly the Vasicek single-factor model evaluated at the 99.9% quantile. My project computes both the analytical and the simulated version and they agree to within 1.6%, confirming the implementation.

### Q10. What's Credit VaR and how does it relate to Economic Capital?

Credit VaR is the loss exceeded with probability α on the portfolio loss distribution. Basel uses α = 99.9% for economic capital purposes.

Economic Capital = Credit VaR − Expected Loss. EL is already provisioned through earnings (loan-loss reserves). EC is the unexpected loss portion that requires equity backing.

For my 1,000-obligor sub-prime portfolio: EL = $2.76M, Credit VaR at 99.9% = $9.20M (under t-copula), Economic Capital = $6.44M. The bank would need to hold ~$6.4M of equity against $15M of loans (43% capital-to-loan ratio), which is why sub-prime lending is typically funded outside the regulated banking system.

### Q11. Why FastAPI over Flask for this kind of platform?

Three reasons.

1. **Type-safe request/response via Pydantic.** FastAPI uses Python type hints to generate Pydantic schemas, which validate inputs and serialize outputs automatically. Flask requires you to write that boilerplate yourself.
2. **Auto-generated OpenAPI/Swagger docs at /docs.** This is what makes the API self-documenting and demo-able. Flask has third-party add-ons; FastAPI ships it.
3. **Async-first.** Built on Starlette and ASGI. For I/O-bound risk endpoints (database lookups, model loading, cache hits), async lets a single worker handle many concurrent requests without threading complexity.

A fourth reason is industry direction: most new Python microservices at banks are FastAPI. Flask is legacy.

### Q12. How would you scale this platform to 10 million loans?

The current design scores loans one DataFrame at a time, in-process. To scale 4-5 orders of magnitude:

1. **Move heavy compute out of the request path.** Score loans in background batch jobs (Celery / Dask / Spark), persist results in a database. The API serves cached scores, not live model inference.
2. **Database for portfolios.** Replace the in-memory DataFrames with PostgreSQL (loan-level tables, daily score history, vintage roll-ups). Postgres handles 10M rows without breaking a sweat.
3. **Model serving via TorchServe / Triton / Ray Serve.** Load models once, serve inference over RPC. Decouple model versioning from API deployment.
4. **Parquet for analytics.** Daily exports to columnar storage (S3 + Parquet + DuckDB/Athena) for ad-hoc credit analytics.
5. **Horizontal scaling.** The API is stateless; behind a load balancer you can run N copies. Kubernetes or AWS Fargate for orchestration.
6. **Caching.** Score distributions don't change between batch runs; cache aggressively (Redis) for the dashboard's read-heavy endpoints.

The capability boundary at 10M loans isn't the math — that's still PD × LGD × EAD per loan. It's the I/O, deployment, monitoring, and model-lifecycle infrastructure. The math takes 1% of the effort, the platform takes 99%.

---

## 4. Project-specific deep-dives

### "Why LendingClub specifically?"

Three reasons. (1) Real default outcomes are publicly available — most credit datasets are proprietary. (2) 2.2M loans across multiple vintages covers enough history to do time-based splits cleanly. (3) It's an interview-recognized dataset; nearly every credit-risk reviewer has seen LendingClub work, so the conversation starts at "how did you handle X" rather than "what is this data."

The honest weakness is that it's unsecured consumer credit, which means LGD is structurally near 1 with almost no variation to model. The LGD model captures the mean but adds no discriminative signal. For a richer LGD story I'd need secured-credit data (mortgages, autos).

### "Why both scorecard AND XGBoost?"

Because the right credit-risk answer is "use both for different things." Scorecard for the final auditable decision (regulator-friendly, decomposable, stable). XGBoost as a challenger model for portfolio-level risk-based pricing and for ranking power inside the decision boundary. In production you'd use the XGBoost output as a *feature* of the scorecard, not as a replacement.

### "Why Vasicek AND copula MC?"

Vasicek analytical gives the Basel IRB number — what the regulator wants. Copula MC gives you the full loss distribution, which is what you need for portfolio management (Credit VaR by sector, marginal contributions, tail-event scenario analysis). They answer different questions. The fact that they agree at the asymptotic limit is a correctness check on both implementations.

### "Why is your LGD R² near zero?"

Because LGD on unsecured consumer credit is empirically very hard to predict at origination. Recoveries are dominated by post-default events (which collections firm gets the account, whether the borrower files bankruptcy, secondary market sale prices) that aren't observable when the loan is originated. The mean LGD on my LendingClub sample is 0.92 with very tight dispersion. A model that captures the mean (which mine does) is doing about as well as theory allows. Sophisticated banks acknowledge this; many use a calibrated unconditional mean for unsecured LGD rather than spending model-risk-management budget chasing R² that isn't there.

### "What's the biggest weakness of your platform?"

Pick one and defend it.

1. **Portfolio is static.** Real books have time-varying weights and refinance/prepay dynamics. A more rigorous version would treat the portfolio as a time series.
2. **No counterparty / collateral.** Pure unsecured PD/LGD/EAD. Real loan books need collateral haircuts, guarantor netting, sovereign overlays.
3. **No stress-test integration on credit.** Market stress scenarios are wired up; credit stress (forced PD/LGD multipliers like CCAR scenarios) is not.
4. **No model risk uncertainty.** Each PD/LGD is a point estimate. A real model-risk framework would express uncertainty intervals.
5. **The Docker image ships with stub models.** The trained pickles aren't bundled (size + reproducibility concerns); the user must train locally. Documented but a friction point for casual reviewers.

### "How would you extend this?"

1. **Bring in mortgage / secured data** (Freddie Mac) to add LGD signal.
2. **Implement the CCAR / DFAST stress framework**: apply supervisor scenarios to PD and LGD simultaneously, recompute portfolio loss.
3. **Risk-based pricing**: use the PD/LGD output to compute break-even loan rate per borrower.
4. **Sector concentration**: cluster obligors by sector, add sector-level correlations to the copula.
5. **Model challenger pipeline**: A/B framework that automatically scores production loans through both scorecard and XGBoost and tracks the gap over time.

---

## 5. Common traps and what NOT to say

- **Don't say "I used train/test split"** without specifying it was time-based. Random splits in credit are a red flag.
- **Don't conflate AUC and calibration.** AUC measures ranking, calibration measures absolute probability accuracy. A model can have great AUC and terrible calibration (and vice versa). Capital calculations need calibrated PDs, not just ranking.
- **Don't say Vasicek is "different from" Basel IRB.** It IS Basel IRB. The capital formula and the Vasicek single-factor model are the same math.
- **Don't say copulas "solved" the 2008 problem.** Copulas were *part of* the 2008 problem. The post-2008 fix was multi-faceted: t-copula or more, stress testing, higher capital ratios, CCAR, model risk management frameworks (SR 11-7).
- **Don't claim FastAPI handles concurrency "automatically."** It handles async/await well, but you still have to think about thread-safety of shared state (loaded models, DB sessions, caches).
- **Don't say PSI < 0.10 means the model is healthy.** It means the score *distribution* is stable. The model could still be miscalibrated. Both must be monitored.

---

## 6. Cheat sheet — the actual numbers

**PD scorecard (test, 2018 vintage):**
- AUC: 0.693, KS: 0.284, Gini: 0.386
- PSI train -> test: 0.030 (stable)
- Top features by IV: int_rate 0.51, term 0.21, fico 0.12, dti 0.08

**PD XGBoost (test):**
- AUC: 0.713, KS: 0.314, Gini: 0.426
- XGBoost edge over scorecard: +2.0 AUC pp

**LGD (test on defaulted loans):**
- Mean realized: 0.92 (mean predicted matches)
- MAE: 0.07
- R² near zero (structural, not a bug)

**Portfolio credit (1000 obligors, PD 20%, LGD 92%, ρ 0.10, $15M EAD):**
- Vasicek ASRF: EL $2.76M, VaR 99.9% $7.68M, EC $4.92M, cond. PD at 99.9% = 55.7%
- Gaussian MC (100k sims): VaR 99.9% $7.81M, EC $5.05M
- **Student-t MC (df=5): VaR 99.9% $9.20M, EC $6.44M (28% above Gaussian)**

**Sample portfolio EL (50 LendingClub loans, $750K EAD):**
- Weighted PD: 23%, weighted LGD: 92%
- Total EL: ~21% of EAD
- RWA density: ~459% (37% capital-to-loan ratio)

**Market risk backtests (2014-2026):**
- Plain Historical / Parametric-Normal / MC-Normal: all fail Christoffersen at 95% (p ≈ 0.000)
- Parametric-Normal 99% exceedance rate: 2.41% (vs nominal 1%) — Kupiec p = 0.000
- **FHS GARCH(1,1) is the only method to pass both Kupiec and Christoffersen at both 95% and 99%**

**EVT (POT-GPD on 5% worst losses):**
- ξ = 0.31 (heavy tail confirmed)
- 99.9% VaR = 4.14% (vs Parametric-Normal ~2.05%, 2× higher)

---

## 7. Drill plan

- **Day 1:** Read this whole document. Highlight sentences you can't say without looking.
- **Day 2:** Practice the elevator pitch out loud 10 times. Record yourself once and listen back.
- **Day 3:** Answer Q1-Q12 out loud, untimed.
- **Day 4:** Re-do the fumbled ones, then all 12 timed (60 seconds each).
- **Day 5:** Project-specific deep dives. The "biggest weakness" answer is the one interviewers probe — practice it especially.
- **Day 6:** Have someone quiz you in random order. No notes visible.
- **Day 7:** Rest before the interview.

Hard rules:
1. **Speak the answers out loud,** in your head doesn't count. Speech production is the bottleneck under stress.
2. **Use specific numbers.** "FHS passes with Kupiec p = 0.55" is much more credible than "FHS passes."
3. **If you don't know, say so.** "I'm not certain, but my best guess is X because Y" is much better than confidently bullshitting. Risk people are trained to detect overconfidence.
