"""Phase 0 integration tests for the FastAPI app.

Uses FastAPI's TestClient (requires httpx). No Docker needed for these.
"""

from fastapi.testclient import TestClient

from risk_platform.api.main import app


client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_version_includes_models():
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert "platform_version" in body
    assert "models" in body and "scorecard_pd" in body["models"]


def test_market_var_endpoint():
    r = client.post("/market/var",
                    json={"method": "historical", "alpha": 0.05})
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["VaR"] < 1
    assert body["method"] == "historical"


SAMPLE_LOAN = {
    "loan_amnt": 15_000, "int_rate": 10.5, "term": "36 months",
    "annual_inc": 80_000, "fico": 700, "dti": 18.0,
    # everything else uses schema defaults
}


def test_credit_pd_endpoint():
    r = client.post("/credit/pd", json={"loan": SAMPLE_LOAN, "model": "scorecard"})
    assert r.status_code == 200
    body = r.json()
    assert 0 < body["pd"] < 1


def test_expected_loss_endpoint():
    r = client.post("/credit/expected_loss?pd_model=scorecard", json=SAMPLE_LOAN)
    assert r.status_code == 200
    body = r.json()
    assert body["expected_loss"] > 0
    assert body["rwa"] > 0


def test_portfolio_credit_var_endpoint():
    r = client.post("/portfolio/credit_var",
                    json={"pd": 0.05, "rho": 0.15, "copula": "t", "df": 5})
    assert r.status_code == 200
    assert r.json()["copula"] == "t"


def test_unified_risk_report():
    body = {"loan": SAMPLE_LOAN,
            "market_method": "historical", "market_alpha": 0.05,
            "portfolio_pd": 0.05, "portfolio_rho": 0.15}
    r = client.post("/risk_report", json=body)
    assert r.status_code == 200
    out = r.json()
    assert {"market", "loan_expected_loss", "portfolio_credit"} == set(out)
