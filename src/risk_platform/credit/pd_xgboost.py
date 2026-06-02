"""XGBoost PD model.

Companion to the WOE scorecard. XGBoost typically delivers a higher raw AUC
on LendingClub data (handles non-linearities and feature interactions that a
linear model misses), at the cost of interpretability. We use SHAP values for
the explainability layer.

This implements the same public interface as ScorecardPD so the API/dashboard
endpoints work for either model interchangeably.
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
class XGBoostPD:
    """Gradient-boosted PD with native categorical support."""

    version: str = "xgboost-0.1.0"

    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    model: xgb.XGBClassifier | None = None
    feature_cols: list[str] = field(default_factory=list)

    # Scorecard-style PDO scoring (so we can produce a 600-style score too)
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    # ----------------------------------------------------------------------
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cast categoricals to pandas Categorical for XGBoost's native handling."""
        X = X[self.feature_cols].copy()
        for col in self.categorical_features:
            if col in X.columns:
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
        params: dict[str, Any] | None = None,
        n_estimators: int = 600,
        early_stopping_rounds: int = 30,
    ) -> "XGBoostPD":
        """Fit XGBoost. If X_val/y_val are given, early-stops on validation logloss."""
        self.numeric_features = numeric_features or self.numeric_features
        self.categorical_features = categorical_features or self.categorical_features
        self.feature_cols = self.numeric_features + self.categorical_features

        # Sensible defaults; tuned set will land in Phase 1b via optuna.
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "enable_categorical": True,
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_weight": 50,    # large enough to fight overfit on 900k rows
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "n_estimators": n_estimators,
            "early_stopping_rounds": early_stopping_rounds if X_val is not None else None,
            "random_state": 42,
            "n_jobs": -1,
        }
        if params:
            default_params.update(params)

        self.model = xgb.XGBClassifier(**{k: v for k, v in default_params.items() if v is not None})
        Xp = self._prepare(X)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(self._prepare(X_val), y_val)]
        self.model.fit(Xp, y, eval_set=eval_set, verbose=False)
        return self

    # ----------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame | dict[str, Any]) -> np.ndarray:
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        return self.model.predict_proba(self._prepare(X))[:, 1]

    def score(self, X: pd.DataFrame | dict[str, Any]) -> np.ndarray:
        """600/PDO=20 style score from the predicted PD."""
        from .pd_scorecard import _logodds_to_score  # reuse helper
        p = np.clip(self.predict_proba(X), 1e-6, 1 - 1e-6)
        log_odds = np.log(p / (1 - p))
        return _logodds_to_score(log_odds, self.base_score, self.base_odds, self.pdo).round().astype(int)

    # ----------------------------------------------------------------------
    def feature_importance(self, kind: str = "gain") -> pd.DataFrame:
        """Built-in XGBoost importance ('gain', 'weight', or 'cover')."""
        booster = self.model.get_booster()
        score = booster.get_score(importance_type=kind)
        # Booster maps feature index strings like f0 to actual names via booster.feature_names
        names = booster.feature_names or self.feature_cols
        rows = [(names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k, v)
                for k, v in score.items()]
        return (
            pd.DataFrame(rows, columns=["feature", "importance"])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def shap_values(self, X: pd.DataFrame, n_sample: int = 5000) -> tuple[np.ndarray, pd.DataFrame]:
        """Return SHAP values on a random sample of `n_sample` rows.

        Lazily imports shap so it isn't a runtime dependency for callers that
        only use predict_proba. Returns (shap_values, X_used).
        """
        import shap  # noqa: F401 — lazy
        X_sample = X.sample(min(n_sample, len(X)), random_state=42)
        explainer = shap.TreeExplainer(self.model)
        Xp = self._prepare(X_sample)
        sv = explainer.shap_values(Xp)
        return sv, X_sample

    # ----------------------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "XGBoostPD":
        with open(path, "rb") as f:
            return pickle.load(f)
