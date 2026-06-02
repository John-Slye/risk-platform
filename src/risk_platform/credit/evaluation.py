"""PD model evaluation utilities: AUC, KS, Brier, calibration, PSI.

Used for both the scorecard and the XGBoost PD model so we can produce
the side-by-side comparison report Phase 1 calls for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, roc_curve,
)


@dataclass
class EvalResults:
    """Container for one model x one dataset evaluation."""
    auc: float
    ks: float
    gini: float
    brier: float
    base_rate: float
    n_obs: int

    def to_dict(self) -> dict:
        return {
            "AUC": round(self.auc, 4),
            "KS": round(self.ks, 4),
            "Gini": round(self.gini, 4),
            "Brier": round(self.brier, 5),
            "base_rate": round(self.base_rate, 4),
            "n_obs": self.n_obs,
        }


def ks_statistic(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between cumulative
    distributions of predicted probability for goods vs bads."""
    y_true = np.asarray(y_true)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_proba) -> EvalResults:
    """Standard discrimination + calibration metrics on one dataset."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    auc = roc_auc_score(y_true, y_proba)
    ks = ks_statistic(y_true, y_proba)
    return EvalResults(
        auc=auc,
        ks=ks,
        gini=2 * auc - 1,
        brier=brier_score_loss(y_true, y_proba),
        base_rate=float(y_true.mean()),
        n_obs=len(y_true),
    )


def calibration_table(
    y_true, y_proba, n_bins: int = 10
) -> pd.DataFrame:
    """Decile calibration: in each bucket of predicted PD, what is the
    realized default rate? Perfect calibration -> predicted ~= realized.

    Returns a DataFrame with bin, n, mean_predicted, mean_realized,
    suitable for plotting a calibration curve.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bins = pd.qcut(y_proba, q=n_bins, duplicates="drop")
    df = pd.DataFrame({"y": y_true, "p": y_proba, "bin": bins})
    out = (
        df.groupby("bin", observed=True)
        .agg(n=("y", "size"),
             mean_predicted=("p", "mean"),
             mean_realized=("y", "mean"))
        .reset_index(drop=True)
    )
    out["error_pp"] = (out["mean_realized"] - out["mean_predicted"]) * 100
    return out
