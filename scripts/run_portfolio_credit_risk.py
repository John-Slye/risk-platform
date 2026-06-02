"""Phase 3 demo: Vasicek + Gaussian/t copula portfolio loss distributions.

Run from project root:
    uv run python scripts/run_portfolio_credit_risk.py

Produces the headline figure of Phase 3:
    figures/portfolio_credit_loss_distribution.png

Plus prints a side-by-side Vasicek-analytical / Gaussian-MC / t-MC table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_platform.portfolio import (
    asrf_loss_distribution, simulate_portfolio_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


def main() -> None:
    # Portfolio assumptions — sub-prime consumer book matching Phase 1/2 numbers.
    pd = 0.20
    lgd = 0.92
    ead = 15_000.0       # $ per loan
    rho = 0.10           # asset correlation; retail Basel ~0.04, sub-prime ~0.10-0.15
    n_obligors = 1_000
    n_sims = 100_000

    print(f"Portfolio: {n_obligors:,} obligors x ${ead:,.0f} EAD each = "
          f"${n_obligors * ead:,.0f} total")
    print(f"Per-obligor PD = {pd*100:.1f}%, LGD = {lgd*100:.1f}%, rho = {rho}")
    print()

    # ----------- Vasicek analytical (Basel IRB foundation) -----------
    asrf = asrf_loss_distribution(pd, rho, lgd, ead, n_obligors)
    print("--- Vasicek ASRF (large-portfolio analytical) ---")
    print(f"  Expected Loss   : ${asrf['expected_loss']:>14,.0f}")
    print(f"  Credit VaR 99%  : ${asrf['VaR_99.0']:>14,.0f}")
    print(f"  Credit VaR 99.9%: ${asrf['VaR_99.9']:>14,.0f}")
    print(f"  Economic Capital: ${asrf['economic_capital_99_9']:>14,.0f}")
    print(f"  Cond. PD @ 99.9%: {asrf['cond_pd_99.9']*100:>14.2f}%   (vs unconditional {pd*100:.1f}%)")

    # ----------- Gaussian copula Monte Carlo -----------
    print(f"\n--- Gaussian copula MC ({n_sims:,} sims) ---")
    g = simulate_portfolio_loss(pd, ead, lgd, rho, n_obligors, n_sims, "gaussian")
    print(f"  Expected Loss   : ${g['expected_loss']:>14,.0f}")
    print(f"  Credit VaR 99%  : ${g['credit_var_99']:>14,.0f}")
    print(f"  Credit VaR 99.9%: ${g['credit_var_99_9']:>14,.0f}")
    print(f"  Economic Capital: ${g['economic_capital']:>14,.0f}")
    print(f"  Tail ES 99.9%   : ${g['tail_es_99_9']:>14,.0f}")

    # ----------- Student-t copula Monte Carlo -----------
    df = 5
    print(f"\n--- Student-t copula MC, df={df} ({n_sims:,} sims) ---")
    t = simulate_portfolio_loss(pd, ead, lgd, rho, n_obligors, n_sims, "t", df=df)
    print(f"  Expected Loss   : ${t['expected_loss']:>14,.0f}")
    print(f"  Credit VaR 99%  : ${t['credit_var_99']:>14,.0f}")
    print(f"  Credit VaR 99.9%: ${t['credit_var_99_9']:>14,.0f}")
    print(f"  Economic Capital: ${t['economic_capital']:>14,.0f}")
    print(f"  Tail ES 99.9%   : ${t['tail_es_99_9']:>14,.0f}")

    print("\n--- t vs Gaussian tail ratio ---")
    print(f"  Credit VaR 99.9% : t / Gauss = {t['credit_var_99_9'] / g['credit_var_99_9']:.2f}x")
    print(f"  Economic Capital : t / Gauss = {t['economic_capital'] / g['economic_capital']:.2f}x")

    # ----------- Money-shot chart -----------
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = np.linspace(0, max(g["losses"].max(), t["losses"].max()) * 1.02, 80)
    ax.hist(g["losses"], bins=bins, color="steelblue", alpha=0.55, label="Gaussian copula")
    ax.hist(t["losses"], bins=bins, color="crimson",   alpha=0.55, label=f"Student-t copula (df={df})")
    ax.axvline(g["expected_loss"], color="black",     ls="--", lw=1, label=f"EL = ${g['expected_loss']:,.0f}")
    ax.axvline(g["credit_var_99_9"], color="steelblue", ls=":", lw=2, label=f"Gauss 99.9% VaR = ${g['credit_var_99_9']:,.0f}")
    ax.axvline(t["credit_var_99_9"], color="crimson",   ls=":", lw=2, label=f"t     99.9% VaR = ${t['credit_var_99_9']:,.0f}")
    ax.set_xlabel("Portfolio loss ($)")
    ax.set_ylabel("Simulation count")
    ax.set_title(
        f"Portfolio loss distribution (1k obligors, PD={pd*100:.0f}%, "
        f"LGD={lgd*100:.0f}%, rho={rho})"
    )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "portfolio_credit_loss_distribution.png", dpi=140)
    plt.close(fig)
    print(f"\nWrote {FIG_DIR/'portfolio_credit_loss_distribution.png'}")


if __name__ == "__main__":
    main()
