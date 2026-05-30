"""Probability of Default models.

Phase 0 stubs: return a fixed PD so the API/dashboard pipeline can be wired up
end-to-end. Phase 1 replaces these with the real scorecard and XGBoost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class _StubPDModel:
    """Base for stub PD models. Returns a hardcoded probability."""
    version: str = "stub-0.1.0"
    _hardcoded_pd: float = 0.05

    def predict_proba(self, features: dict[str, Any]) -> float:
        """Return PD given a dict of loan features. Phase 0 ignores inputs."""
        return self._hardcoded_pd


class ScorecardPD(_StubPDModel):
    """Logistic-regression scorecard with WOE/IV binning.

    Phase 1 will implement:
      - optbinning.BinningProcess for WOE/IV feature engineering
      - statsmodels logistic regression on WOE features
      - Conversion to 300-850 style score via PDO scaling
      - PSI for monitoring score drift
    """
    version: str = "scorecard-stub-0.1.0"
    _hardcoded_pd: float = 0.045

    def score(self, features: dict[str, Any]) -> int:
        """Return a 300-850 style credit score. Hardcoded in Phase 0."""
        return 720


class XGBoostPD(_StubPDModel):
    """Gradient-boosted PD model.

    Phase 1 will implement:
      - xgboost.XGBClassifier with optuna hyperparameter search
      - Time-based train/validate/test splits
      - SHAP value computation for interpretability
      - Platt scaling or isotonic calibration to align ranking with probabilities
    """
    version: str = "xgboost-stub-0.1.0"
    _hardcoded_pd: float = 0.052
