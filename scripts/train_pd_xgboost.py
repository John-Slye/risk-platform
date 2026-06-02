"""Train XGBoost PD model on LendingClub, with early stopping on val set.

Run from project root:
    uv run python scripts/train_pd_xgboost.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk_platform.credit.evaluation import EvalResults, calibration_table, evaluate
from risk_platform.credit.pd_xgboost import XGBoostPD
from risk_platform.data.lending_club import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_processed, time_split,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pd_xgboost.pkl"


def main() -> None:
    print("Loading processed data...")
    df = load_processed()
    train, val, test = time_split(df)
    print(f"  train: {len(train):>9,} (default rate {train['default'].mean()*100:.2f}%)")
    print(f"  val  : {len(val):>9,} (default rate {val['default'].mean()*100:.2f}%)")
    print(f"  test : {len(test):>9,} (default rate {test['default'].mean()*100:.2f}%)")

    numeric = NUMERIC_FEATURES
    categorical = CATEGORICAL_FEATURES

    X_train, y_train = train[numeric + categorical], train["default"]
    X_val,   y_val   = val[numeric + categorical],   val["default"]
    X_test,  y_test  = test[numeric + categorical],  test["default"]

    print("\nFitting XGBoost (early stopping on val, ~30-60 sec)...")
    m = XGBoostPD()
    m.fit(X_train, y_train, X_val=X_val, y_val=y_val,
          numeric_features=numeric, categorical_features=categorical)
    print(f"Best iteration: {m.model.best_iteration if hasattr(m.model, 'best_iteration') else 'n/a'}")

    print("\nTop 15 features by gain:")
    print(m.feature_importance("gain").head(15).to_string(index=False))

    print("\n========== Evaluation ==========")
    for name, X, y in [("Train", X_train, y_train), ("Val ", X_val, y_val), ("Test", X_test, y_test)]:
        proba = m.predict_proba(X)
        print(f"{name}: {evaluate(y, proba).to_dict()}")

    proba_test = m.predict_proba(X_test)
    cal = calibration_table(y_test.values, proba_test, n_bins=10)
    print("\nTest set decile calibration (predicted vs realized PD):")
    print(cal.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    m.save(MODEL_PATH)
    print(f"\nSaved fitted XGBoost to {MODEL_PATH}")


if __name__ == "__main__":
    main()
