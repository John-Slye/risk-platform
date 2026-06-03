# Demo Walkthrough

Step-by-step click-through someone can follow without watching a video. Roughly 3 minutes end-to-end.

## Prerequisites

```bash
docker compose up --build
```

Wait for the log line `risk-platform-dashboard | You can now view your Streamlit app...` before proceeding.

Open in two browser tabs:
- http://localhost:8000/docs (FastAPI Swagger UI)
- http://localhost:8501 (Streamlit dashboard)

---

## 1. The API (~30 seconds)

Tab: http://localhost:8000/docs

1. Scroll to **POST /credit/pd**, click "Try it out."
2. In the request body, replace the defaults with:
   ```json
   {
     "loan": {
       "loan_amnt": 15000,
       "int_rate": 10.5,
       "term": "36 months",
       "annual_inc": 80000,
       "fico": 700,
       "dti": 18.0
     },
     "model": "scorecard"
   }
   ```
3. Click "Execute." You get back a PD, a credit score, and the model version. **The model version tells you whether the real trained model or the stub is answering** — useful for production debugging.

---

## 2. Market risk page (~30 seconds)

Tab: http://localhost:8501, sidebar -> **Market Risk**.

1. Confidence dropdown: **99%**.
2. Click "**Compute all methods at 99%**." After a few seconds you'll see a Plotly bar chart showing all 6 methods side by side.
3. Read off the chart: Parametric-Normal and MC-Normal are visibly lower than Historical and the t-distribution methods. **That gap is the fat-tail underestimate** the project is built to expose.

---

## 3. Portfolio credit risk page (~30 seconds)

Sidebar -> **Portfolio Credit**.

1. Form defaults (PD 20%, LGD 92%, ρ 0.10, 1000 obligors).
2. Copula: **t**, df 5.
3. Click "Simulate portfolio." After ~5 seconds you see:
   - Expected Loss ~$2.76M (dashed line on the chart)
   - 99% Credit VaR ~$7.4M (orange line)
   - 99.9% Credit VaR ~$9.2M (red line)
   - Economic Capital = 99.9% VaR minus EL = ~$6.4M
4. Re-run with copula = **gaussian** and compare: VaR 99.9% drops to ~$7.8M, EC to ~$5.0M. **The 28% gap** is the post-2008 modeling story.

---

## 4. Portfolio upload page (~60 seconds)

Sidebar -> **Portfolio Upload**.

1. Click "**Download sample 50-loan CSV**." Save to your Downloads folder.
2. Drag-drop the downloaded file into the uploader.
3. Click "**Run portfolio analysis**." After ~10 seconds you see:
   - Aggregate Expected Loss, total RWA, weighted PD, weighted LGD
   - Top 10 riskiest loans by EL
   - PD vs LGD scatter (size = EAD, color = EL)
   - Download button for the per-loan results CSV

This is what a credit officer would see for a real loan book.

---

## 5. Talk track if you're walking someone through

When a reviewer asks "tell me about this," answer in this order:

1. "It's an integrated credit and market risk platform - everything from PD modeling to portfolio Credit VaR behind one FastAPI backend and a Streamlit dashboard, all in Docker."
2. "The credit headline finding is that the Student-t copula adds 28% to Economic Capital versus Gaussian on a sub-prime portfolio - the central post-2008 modeling story."
3. "The market headline finding is that of five VaR methods, only Filtered Historical Simulation with GARCH passes both Kupiec and Christoffersen backtests at 95% and 99%."
4. "I built the market engine first as a standalone project, then ported it in as a module - that integration was Phase 4 of the build."
5. "Numbers are in the README. Methodology PDF is in /docs. CI is green on every commit."

That's the 60-second pitch when you're sharing the screen.

---

## Stopping

```bash
# Ctrl-C in the terminal running docker compose, then:
docker compose down
```
