"""Logistic-regression scorecard PD model with WOE/IV binning.

This module implements the real scorecard. It replaces the stub in
`pd_models.py` (which we keep for Phase 0 testing). The class signature
remains compatible so the FastAPI endpoint and dashboard pages do not
need to change.

Workflow:
    sc = ScorecardPD()
    sc.fit(X_train, y_train)
    sc.predict_proba(X_test)   # -> probability of default in [0, 1]
    sc.score(X_test)           # -> 300-850 style integer score
    sc.feature_iv()            # -> DataFrame of feature IV values
    sc.psi(X_ref, X_new)       # -> Population Stability Index
    sc.save(path); ScorecardPD.load(path)

Math conventions:
  - WOE_bin = ln( (goods_in_bin/total_goods) / (bads_in_bin/total_bads) )
    Higher WOE means lower default risk.
  - IV     = sum over bins of (goods% - bads%) * WOE
  - PDO scoring: base_score 600 at base_odds 50, PDO=20 (industry standard).
    Higher score = lower risk.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from optbinning import BinningProcess


# --------------------------------------------------------------------------
# Helpers: PDO scoring
# --------------------------------------------------------------------------
def _logodds_to_score(
    log_odds: np.ndarray, base_score: float = 600.0,
    base_odds: float = 50.0, pdo: float = 20.0,
) -> np.ndarray:
    """Convert log-odds (of good : bad) to a credit score.

    PDO formula: score = base_score - factor * log(odds_of_bad)
      where  factor = pdo / ln(2)
             offset = base_score - factor * ln(base_odds)
    We accept `log_odds` here as ln(P(bad)/P(good)) — i.e. logit of default
    probability. Higher logit -> higher PD -> lower score.
    """
    factor = pdo / math.log(2)
    offset = base_score + factor * math.log(base_odds)
    return offset - factor * log_odds


# --------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------
@dataclass
class ScorecardPD:
    """WOE + logistic-regression scorecard.

    Default feature lists match the LendingClub data module; override via
    `numeric_features` / `categorical_features` for other datasets.
    """

    version: str = "scorecard-0.1.0"

    # Scoring constants (PDO convention)
    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    # Feature lists set at fit() time
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)

    # Internals (populated by fit)
    binning_process: BinningProcess | None = None
    model: Any = None              # statsmodels.GLM result
    feature_cols: list[str] = field(default_factory=list)
    iv_table: pd.DataFrame | None = None
    train_score_dist: pd.Series | None = None

    # ----------------------------------------------------------------------
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        min_iv: float = 0.02,
        max_n_bins: int = 8,
    ) -> "ScorecardPD":
        """Fit binning + logistic regression on WOE-transformed features.

        Steps:
            1. optbinning.BinningProcess finds optimal monotonic bins per
               numeric feature, and grouped bins per categorical.
            2. Drop features with IV < min_iv (weak signal).
            3. Transform all features to WOE.
            4. Fit statsmodels.Logit on WOE features (gives p-values, etc).
        """
        self.numeric_features = numeric_features or list(self.numeric_features)
        self.categorical_features = categorical_features or list(self.categorical_features)
        if not self.numeric_features and not self.categorical_features:
            # Infer from X dtypes
            self.numeric_features = X.select_dtypes(include="number").columns.tolist()
            self.categorical_features = [c for c in X.columns if c not in self.numeric_features]

        all_features = self.numeric_features + self.categorical_features

        # Step 1: optimal binning. selection_criteria filters weak features.
        self.binning_process = BinningProcess(
            variable_names=all_features,
            categorical_variables=self.categorical_features,
            max_n_bins=max_n_bins,
            min_bin_size=0.02,
            selection_criteria={"iv": {"min": min_iv, "max": 1.0}},
        )
        self.binning_process.fit(X[all_features], y.values)

        # Information Value table (sorted descending)
        info_df = self.binning_process.summary().copy()
        name_col = "name" if "name" in info_df.columns else info_df.columns[0]
        info_df = info_df.rename(columns={name_col: "feature"})
        keep_cols = [c for c in
                     ["feature", "iv", "n_bins", "status", "selected"]
                     if c in info_df.columns]
        self.iv_table = info_df[keep_cols].sort_values(
            "iv", ascending=False
        ).reset_index(drop=True)

        # Step 2: keep features that survived the IV selection_criteria above.
        # optbinning's `summary()` has a boolean "selected" column for this.
        if "selected" in info_df.columns:
            kept = info_df[info_df["selected"].astype(bool)]["feature"].tolist()
        else:
            # Older optbinning: fall back to get_support
            kept = list(self.binning_process.get_support(names=True))
        if not kept:
            raise RuntimeError(
                "No features survived IV selection. Lower min_iv or check data."
            )
        self.feature_cols = kept

        # Step 3: WOE transform
        X_woe = self.binning_process.transform(X[all_features], metric="woe")
        X_woe = X_woe[kept]

        # Step 4: logistic regression with statsmodels (gives stderr + pvalues)
        X_sm = sm.add_constant(X_woe, has_constant="add")
        self.model = sm.Logit(y.values, X_sm).fit(disp=False, maxiter=100)

        # Cache training-set score distribution for PSI
        self.train_score_dist = pd.Series(self.score(X))

        return self

    # ----------------------------------------------------------------------
    def _to_woe(self, X: pd.DataFrame) -> pd.DataFrame:
        all_features = self.numeric_features + self.categorical_features
        X_woe = self.binning_process.transform(X[all_features], metric="woe")
        return X_woe[self.feature_cols]

    def predict_proba(self, X: pd.DataFrame | dict[str, Any]) -> np.ndarray:
        """Predict probability of default in [0, 1]. Accepts dict for one loan."""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X_woe = self._to_woe(X)
        X_sm = sm.add_constant(X_woe, has_constant="add")
        return self.model.predict(X_sm)

    def predict_logodds(self, X: pd.DataFrame | dict[str, Any]) -> np.ndarray:
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X_woe = self._to_woe(X)
        X_sm = sm.add_constant(X_woe, has_constant="add")
        # Logit of default probability
        return X_sm.values @ self.model.params.values

    def score(self, X: pd.DataFrame | dict[str, Any]) -> np.ndarray:
        """Convert log-odds to integer credit score under PDO convention."""
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        log_odds = self.predict_logodds(X)
        return _logodds_to_score(
            log_odds, self.base_score, self.base_odds, self.pdo
        ).round().astype(int)

    # ----------------------------------------------------------------------
    def feature_iv(self) -> pd.DataFrame:
        """Information Values for every candidate feature (incl. dropped)."""
        return self.iv_table.copy()

    def psi(
        self, X_ref: pd.DataFrame, X_new: pd.DataFrame, n_bins: int = 10
    ) -> dict[str, float]:
        """Population Stability Index between two score distributions.

        Bins the *reference* score distribution into n_bins quantiles, then
        compares fraction in each bin on the new dataset.

        PSI < 0.10: stable. 0.10-0.25: monitor. > 0.25: retrain.
        """
        ref_scores = self.score(X_ref)
        new_scores = self.score(X_new)

        # Quantile bin edges from reference
        edges = np.quantile(ref_scores, np.linspace(0, 1, n_bins + 1))
        edges[0] -= 1
        edges[-1] += 1
        ref_pct, _ = np.histogram(ref_scores, bins=edges)
        new_pct, _ = np.histogram(new_scores, bins=edges)
        ref_pct = ref_pct / ref_pct.sum()
        new_pct = new_pct / new_pct.sum()

        # Avoid log(0) via floor
        eps = 1e-6
        ref_pct = np.clip(ref_pct, eps, 1)
        new_pct = np.clip(new_pct, eps, 1)
        psi_per_bin = (new_pct - ref_pct) * np.log(new_pct / ref_pct)
        return {
            "psi": float(psi_per_bin.sum()),
            "interpretation": (
                "stable" if psi_per_bin.sum() < 0.10
                else "monitor" if psi_per_bin.sum() < 0.25
                else "retrain"
            ),
            "n_bins": n_bins,
        }

    # ----------------------------------------------------------------------
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ScorecardPD":
        with open(path, "rb") as f:
            return pickle.load(f)
