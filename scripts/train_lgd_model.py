"""Train LGD regressor on LendingClub defaulted loans only.

Run from project root:
    uv run python scripts/train_lgd_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from risk_platform.credit.lgd_model import LGDModel
from risk_platform.data.lending_club import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_processed, time_split,
)

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pd_lgd.pkl"


def main() -> None:
    print("Loading processed data...")
    df = load_processed()
    train, val, test = time_split(df)

    # LGD is defined only for defaulted loans
    train = train[train["default"] == 1].dropna(subset=["realized_lgd"])
    val   = val  [val  ["default"] == 1].dropna(subset=["realized_lgd"])
    test  = test [test ["default"] == 1].dropna(subset=["realized_lgd"])
    print(f"  train (defaults only): {len(train):>7,}  mean LGD {train['realized_lgd'].mean():.3f}")
    print(f"  val   (defaults only): {len(val):>7,}  mean LGD {val['realized_lgd'].mean():.3f}")
    print(f"  test  (defaults only): {len(test):>7,}  mean LGD {test['realized_lgd'].mean():.3f}")

    numeric = NUMERIC_FEATURES
    categorical = CATEGORICAL_FEATURES
    X_train, y_train = train[numeric + categorical], train["realized_lgd"]
    X_val,   y_val   = val  [numeric + categorical], val  ["realized_lgd"]
    X_test,  y_test  = test [numeric + categorical], test ["realized_lgd"]

    print("\nFitting LGD regressor (XGBoost, ~30 sec)...")
    m = LGDModel()
    m.fit(X_train, y_train, X_val=X_val, y_val=y_val,
          numeric_features=numeric, categorical_features=categorical)
    print(f"Best iteration: {getattr(m.model, 'best_iteration', 'n/a')}")

    print("\n========== Evaluation ==========")
    for name, X, y in [("Train", X_train, y_train), ("Val ", X_val, y_val), ("Test", X_test, y_test)]:
        pred = m.predict_batch(X)
        print(f"{name}: MAE = {mean_absolute_error(y, pred):.4f}, "
              f"R² = {r2_score(y, pred):+.4f}, "
              f"mean_pred = {pred.mean():.3f}, mean_actual = {y.mean():.3f}, n = {len(y):,}")

    # Sanity decile table on test
    print("\nTest decile comparison (predicted vs realized LGD):")
    pred_t = m.predict_batch(X_test)
    dec = pd.qcut(pred_t, q=10, duplicates="drop")
    out = (pd.DataFrame({"pred": pred_t, "actual": y_test.values, "bin": dec})
           .groupby("bin", observed=True)
           .agg(n=("actual", "size"),
                mean_pred=("pred", "mean"),
                mean_actual=("actual", "mean"))
           .reset_index(drop=True))
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    m.save(MODEL_PATH)
    print(f"\nSaved fitted LGD model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
