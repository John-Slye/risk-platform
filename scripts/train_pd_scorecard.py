"""Train the WOE/IV scorecard PD model on LendingClub.

Run from project root:
    uv run python scripts/train_pd_scorecard.py

What it does:
    1. Load processed LendingClub data (cached parquet from data.lending_club).
    2. Vintage-based train/val/test split.
    3. Fit ScorecardPD on train.
    4. Evaluate AUC / KS / Brier / Gini on val and test.
    5. Print Information Value table (which features survived selection).
    6. Compute Population Stability Index train -> val and train -> test.
    7. Save fitted model to models/pd_scorecard.pkl.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_platform.credit.evaluation import (
    EvalResults, calibration_table, evaluate,
)
from risk_platform.credit.pd_scorecard import ScorecardPD
from risk_platform.data.lending_club import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_processed, time_split,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pd_scorecard.pkl"


def main() -> None:
    print("Loading processed data...")
    df = load_processed()
    train, val, test = time_split(df)
    print(f"  train: {len(train):>9,} (default rate {train['default'].mean()*100:.2f}%)")
    print(f"  val  : {len(val):>9,} (default rate {val['default'].mean()*100:.2f}%)")
    print(f"  test : {len(test):>9,} (default rate {test['default'].mean()*100:.2f}%)")

    # NUMERIC_FEATURES already includes 'fico'; no need to append.
    numeric = NUMERIC_FEATURES
    categorical = CATEGORICAL_FEATURES

    X_train = train[numeric + categorical]
    y_train = train["default"]

    print("\nFitting scorecard (this takes ~30 sec on 900k rows)...")
    sc = ScorecardPD()
    sc.fit(X_train, y_train,
           numeric_features=numeric, categorical_features=categorical)

    # ----------- Information Value table --------------
    iv = sc.feature_iv()
    print("\nInformation Value per feature (sorted, status=selected means it survived):")
    print(iv.to_string(index=False))

    # ----------- Evaluation ----------------------------
    def eval_set(name: str, X: pd.DataFrame, y: pd.Series) -> EvalResults:
        proba = sc.predict_proba(X)
        res = evaluate(y, proba)
        print(f"\n{name:>5s}:  {res.to_dict()}")
        return res

    print("\n========== Evaluation ==========")
    eval_set("Train", X_train, y_train)
    eval_set("Val ",  val[numeric + categorical], val["default"])
    eval_set("Test",  test[numeric + categorical], test["default"])

    # ----------- Calibration (deciles) on test ----------
    proba_test = sc.predict_proba(test[numeric + categorical])
    cal = calibration_table(test["default"].values, proba_test, n_bins=10)
    print("\nTest set decile calibration (predicted vs realized PD):")
    print(cal.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # ----------- PSI -----------------------------------
    psi_val = sc.psi(X_train, val[numeric + categorical])
    psi_test = sc.psi(X_train, test[numeric + categorical])
    print("\nPSI on score distribution:")
    print(f"  train -> val : {psi_val['psi']:.4f}  ({psi_val['interpretation']})")
    print(f"  train -> test: {psi_test['psi']:.4f}  ({psi_test['interpretation']})")

    # ----------- Save model ----------------------------
    sc.save(MODEL_PATH)
    print(f"\nSaved fitted scorecard to {MODEL_PATH}")


if __name__ == "__main__":
    main()
