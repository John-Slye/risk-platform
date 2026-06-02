"""Loss Given Default model.

The LendingClub realized-LGD distribution is heavily concentrated near 1.0
(mean ~0.92, 37% at exactly 1.0) with effectively zero mass at 0. That
justifies a simpler single-stage XGBoost regressor over a two-stage model:
the boundary at 0 doesn't exist empirically, so there's nothing for a stage-1
classifier to add.

Predictions are clipped to [0.05, 0.95] to keep the API output sensible.

This class implements `.predict(loan_dict)` for single-loan inference (so the
FastAPI endpoint stays unchanged) and `.predict_batch(X)` for vectorized
inference over a portfolio DataFrame.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class LGDModel:
    """XGBoost regressor on realized LGD for defaulted loans."""

    version: str = "lgd-0.1.0"
    floor: float = 0.05
    ceiling: float = 0.95

    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    feature_cols: list[str] = field(default_factory=list)
    # Frozen train-time category vocab so predict-time unseen values map to NaN.
    _categories: dict[str, list] = field(default_factory=dict)
    model: xgb.XGBRegressor | None = None
    _stub_lgd: float = 0.92

    # ---------------------------------------------------------------------
    def _prepare(self, X: pd.DataFrame, *, fit: bool = False) -> pd.DataFrame:
        X = X[self.feature_cols].copy()
        for col in self.categorical_features:
            if col not in X.columns:
                continue
            if fit:
                # Capture the training-set categories so predict-time uses the same vocab.
                self._categories[col] = list(X[col].astype("category").cat.categories)
            cats = self._categories.get(col)
            if cats is not None:
                # Unknown values at predict time become NaN; XGBoost handles them.
                X[col] = pd.Categorical(X[col], categories=cats)
            else:
                X[col] = X[col].astype("category")
        return X

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        n_estimators: int = 400,
        early_stopping_rounds: int = 20,
    ) -> "LGDModel":
        self.numeric_features = numeric_features or self.numeric_features
        self.categorical_features = categorical_features or self.categorical_features
        self.feature_cols = self.numeric_features + self.categorical_features

        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            enable_categorical=True,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            n_estimators=n_estimators,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            random_state=42,
            n_jobs=-1,
        )
        # IMPORTANT: prepare train FIRST with fit=True so the vocabulary is
        # captured, then prepare val with the same frozen vocab.
        X_prep = self._prepare(X, fit=True)
        eval_set = [(self._prepare(X_val), y_val)] if X_val is not None else None
        self.model.fit(X_prep, y, eval_set=eval_set, verbose=False)
        return self

    # ---------------------------------------------------------------------
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Vectorized LGD prediction over a DataFrame; clipped to [floor, ceil]."""
        if self.model is None:
            return np.full(len(X), self._stub_lgd)
        raw = self.model.predict(self._prepare(X))
        return np.clip(raw, self.floor, self.ceiling)

    def predict(self, X: pd.DataFrame | dict[str, Any]) -> float:
        """Single-loan LGD prediction. Accepts dict or one-row DataFrame."""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        return float(self.predict_batch(X)[0])

    # ---------------------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "LGDModel":
        with open(path, "rb") as f:
            return pickle.load(f)
