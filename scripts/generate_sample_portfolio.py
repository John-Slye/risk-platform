"""Generate a small bundled CSV of LendingClub loans for the dashboard demo.

Run once locally; the produced CSV is committed to the repo at
`data/sample/loan_portfolio.csv` so anyone cloning the repo can immediately
try the Portfolio Upload page without needing the full LendingClub dataset.
"""

from __future__ import annotations

from pathlib import Path

from risk_platform.data.lending_club import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_processed, time_split,
)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "loan_portfolio.csv"


def main(n: int = 50) -> None:
    df = load_processed()
    _, _, test = time_split(df)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    sample = test[cols].sample(n=n, random_state=7).reset_index(drop=True)
    sample.to_csv(OUT_PATH, index=False)
    print(f"Wrote {n} loans to {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
