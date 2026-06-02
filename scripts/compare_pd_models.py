"""Side-by-side comparison of scorecard vs XGBoost PD models.

Run after both training scripts have populated models/.
    uv run python scripts/compare_pd_models.py

Produces:
  - figures/pd_roc_curve.png         ROC curves for both models on test
  - figures/pd_calibration.png       Predicted vs realized PD by decile, both models
  - figures/pd_score_distribution.png Score histograms on test
  - figures/pd_summary_table.csv     The headline metrics table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from risk_platform.credit.evaluation import calibration_table, evaluate
from risk_platform.credit.pd_scorecard import ScorecardPD
from risk_platform.credit.pd_xgboost import XGBoostPD
from risk_platform.data.lending_club import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_processed, time_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def main() -> None:
    print("Loading data and models...")
    df = load_processed()
    _, val, test = time_split(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    Xv, yv = val[feature_cols],  val["default"].values
    Xt, yt = test[feature_cols], test["default"].values

    sc = ScorecardPD.load(MODELS_DIR / "pd_scorecard.pkl")
    xg = XGBoostPD.load(MODELS_DIR / "pd_xgboost.pkl")

    sc_proba_v = sc.predict_proba(Xv)
    sc_proba_t = sc.predict_proba(Xt)
    xg_proba_v = xg.predict_proba(Xv)
    xg_proba_t = xg.predict_proba(Xt)

    # ----------- summary metrics table -----------
    print("\nSummary metrics:")
    rows = []
    for name, y, p in [
        ("Scorecard - Val",   yv, sc_proba_v),
        ("Scorecard - Test",  yt, sc_proba_t),
        ("XGBoost   - Val",   yv, xg_proba_v),
        ("XGBoost   - Test",  yt, xg_proba_t),
    ]:
        r = evaluate(y, p).to_dict()
        rows.append({"model_split": name, **r})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(FIG_DIR / "pd_summary_table.csv", index=False)
    print(f"\nWrote {FIG_DIR/'pd_summary_table.csv'}")

    # ----------- ROC curve overlay -----------
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p, color in [("Scorecard (test)", sc_proba_t, "steelblue"),
                           ("XGBoost (test)",   xg_proba_t, "crimson")]:
        fpr, tpr, _ = roc_curve(yt, p)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("PD models - ROC on out-of-time test (2018 vintage)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pd_roc_curve.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {FIG_DIR/'pd_roc_curve.png'}")

    # ----------- calibration overlay -----------
    cal_sc = calibration_table(yt, sc_proba_t, n_bins=10)
    cal_xg = calibration_table(yt, xg_proba_t, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, cal_sc["mean_predicted"].max()*1.05],
            [0, cal_sc["mean_predicted"].max()*1.05],
            "k--", lw=1, alpha=0.6, label="Perfect calibration")
    ax.plot(cal_sc["mean_predicted"], cal_sc["mean_realized"], "o-",
            color="steelblue", lw=1.8, label="Scorecard")
    ax.plot(cal_xg["mean_predicted"], cal_xg["mean_realized"], "s-",
            color="crimson", lw=1.8, label="XGBoost")
    ax.set_xlabel("Mean predicted PD by decile")
    ax.set_ylabel("Realized default rate by decile")
    ax.set_title("Calibration on out-of-time test (2018 vintage)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pd_calibration.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {FIG_DIR/'pd_calibration.png'}")

    # ----------- score distribution overlay -----------
    sc_scores = sc.score(Xt)
    xg_scores = xg.score(Xt)
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(min(sc_scores.min(), xg_scores.min()),
                       max(sc_scores.max(), xg_scores.max()), 50)
    ax.hist(sc_scores, bins=bins, alpha=0.55, color="steelblue", label="Scorecard")
    ax.hist(xg_scores, bins=bins, alpha=0.55, color="crimson",   label="XGBoost")
    ax.set_xlabel("Credit score (PDO 600/20)")
    ax.set_ylabel("Number of loans (test)")
    ax.set_title("PD-implied score distributions, test vintage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pd_score_distribution.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {FIG_DIR/'pd_score_distribution.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
