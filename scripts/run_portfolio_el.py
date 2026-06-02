"""Run the portfolio EL aggregator on a sample of LendingClub loans.

Demonstrates the end-to-end credit-loss calculation: takes a portfolio of
loans, runs PD + LGD + RWA per loan, prints aggregate metrics + top-risk loans.
"""

from __future__ import annotations

from pathlib import Path

from risk_platform.credit import portfolio_expected_loss
from risk_platform.credit.lgd_model import LGDModel
from risk_platform.credit.pd_xgboost import XGBoostPD
from risk_platform.data.lending_club import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_processed, time_split,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main(n_loans: int = 1000) -> None:
    print("Loading models + sample portfolio...")
    pd_model = XGBoostPD.load(MODELS_DIR / "pd_xgboost.pkl")
    lgd_model = LGDModel.load(MODELS_DIR / "pd_lgd.pkl")

    df = load_processed()
    _, _, test = time_split(df)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["loan_amnt", "term"]
    # `loan_amnt` and `term` are already in those lists; dedupe.
    cols = list(dict.fromkeys(cols))
    sample = test[cols].sample(n=n_loans, random_state=42).reset_index(drop=True)
    print(f"  Sampled {len(sample):,} loans from test vintage (2018)\n")

    out = portfolio_expected_loss(sample, pd_model, lgd_model)

    print("======= Portfolio totals =======")
    print(f"  N loans         : {out['n_loans']:,}")
    print(f"  Total EAD       : ${out['total_ead']:,.0f}")
    print(f"  Total EL        : ${out['total_el']:,.0f}  ({out['el_pct_of_ead']:.2f}% of EAD)")
    print(f"  Total RWA       : ${out['total_rwa']:,.0f}  ({out['rwa_density']:.1f}% density)")
    print(f"  EAD-wtd PD      : {out['weighted_pd']*100:.2f}%")
    print(f"  EAD-wtd LGD     : {out['weighted_lgd']*100:.2f}%")

    print("\n======= Top 5 riskiest loans (by EL) =======")
    cols_show = ["ead", "pd", "lgd", "el", "rwa"]
    fmt = {"ead": "${:,.0f}", "pd": "{:.3f}", "lgd": "{:.3f}",
           "el": "${:,.0f}", "rwa": "${:,.0f}"}
    top = out["top_risky"][cols_show].copy()
    for c, f in fmt.items():
        top[c] = top[c].map(lambda x: f.format(x))
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
