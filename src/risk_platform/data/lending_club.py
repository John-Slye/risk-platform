"""LendingClub loan-level data loader for the PD model.

Pipeline:
  raw 1.6 GB CSV  -->  filter to terminal-status, 2014-2018 vintages
                  -->  cast types, define target
                  -->  cache to data/processed/lending_club.parquet

Then `load_processed()` is fast (<5 seconds) and `time_split()` returns the
2014-2016 / 2017 / 2018 train / val / test partition.

Target definition (documented in the README):
  - 1 (default): loan_status in {Charged Off, Default}
  - 0 (paid):    loan_status == Fully Paid
  - Excluded: Current, Issued, In Grace Period, Late (X days)
    These have unknown final outcomes and would bias the model.

Feature selection:
  Only origination-time features are retained. Post-origination columns
  (recoveries, payments, last_pymnt_*, etc.) are EXCLUDED because they
  would leak the target.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# Project layout helpers ------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

RAW_CSV: Path = (
    DATA_RAW
    / "accepted_2007_to_2018q4.csv"
    / "accepted_2007_to_2018Q4.csv"
)
PROCESSED_PARQUET: Path = DATA_PROCESSED / "lending_club.parquet"


# Features kept at origination time (no target leakage) ----------------------
# `fico` is the FICO midpoint we synthesize from the raw range columns.
NUMERIC_FEATURES: list[str] = [
    "loan_amnt",
    "int_rate",
    "installment",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "fico",                  # synthesized in prepare() from fico_range_low/high
]
CATEGORICAL_FEATURES: list[str] = [
    "term",
    "emp_length",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
]
# Metadata columns we need but not as model features.
META_COLUMNS: list[str] = ["loan_status", "issue_d", "id"]

# Raw columns read from CSV (a few we need only to build `fico`, then drop).
_RAW_ONLY: list[str] = ["fico_range_low", "fico_range_high"]
KEEP_COLUMNS: list[str] = (
    [c for c in NUMERIC_FEATURES if c != "fico"]
    + _RAW_ONLY
    + CATEGORICAL_FEATURES
    + META_COLUMNS
)


# Status-to-target mapping ---------------------------------------------------
DEFAULT_STATUSES: set[str] = {"Charged Off", "Default"}
GOOD_STATUSES: set[str] = {"Fully Paid"}
TERMINAL_STATUSES: set[str] = DEFAULT_STATUSES | GOOD_STATUSES


def load_raw(path: Path = RAW_CSV, nrows: int | None = None) -> pd.DataFrame:
    """Read the LendingClub CSV with only the columns we need.

    `nrows` is useful for quick smoke tests; pass `nrows=100_000`
    to read a sample instead of the full 2.26M rows.
    """
    return pd.read_csv(
        path,
        usecols=KEEP_COLUMNS,
        nrows=nrows,
        low_memory=False,
    )


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter, type-cast, and add the binary target.

    Steps:
      1. Drop rows missing loan_status or issue_d.
      2. Keep only terminal-status loans (Fully Paid / Charged Off / Default).
      3. Parse issue_d into datetime; derive `vintage_year`.
      4. Filter to 2014..2018 vintages.
      5. Coerce numeric features.
      6. Define target column `default` in {0, 1}.
    """
    df = raw.copy()

    # Drop rows with no status or date
    df = df.dropna(subset=["loan_status", "issue_d"])

    # Filter to terminal status only
    df = df[df["loan_status"].isin(TERMINAL_STATUSES)].copy()

    # Parse issue_d. Examples in raw: "Dec-2015", "Jan-2018".
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"])
    df["vintage_year"] = df["issue_d"].dt.year

    # Vintage filter
    df = df[(df["vintage_year"] >= 2014) & (df["vintage_year"] <= 2018)].copy()

    # int_rate and revol_util arrive as strings like "12.69%" in some rows;
    # coerce.
    for col in ("int_rate", "revol_util"):
        if df[col].dtype == "object":
            df[col] = (
                df[col].astype(str).str.rstrip("%").replace("nan", np.nan)
                .astype(float)
            )

    # Coerce remaining numerics (any stragglers as strings).
    for col in NUMERIC_FEATURES:
        if col == "fico":
            continue  # synthesized below
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # FICO midpoint as a single feature; drop the raw range columns afterward.
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2.0
    df = df.drop(columns=_RAW_ONLY)

    # Target
    df["default"] = df["loan_status"].isin(DEFAULT_STATUSES).astype(int)

    # Final column ordering
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[feature_cols + ["default", "issue_d", "vintage_year", "id"]].reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: Path = PROCESSED_PARQUET) -> None:
    df.to_parquet(path, index=False)


def load_processed(path: Path = PROCESSED_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_processed(nrows: int | None = None, refresh: bool = False) -> pd.DataFrame:
    """Convenience: load_raw -> prepare -> save_processed -> return df.

    If the cached parquet exists and `refresh=False`, just load that.
    """
    if PROCESSED_PARQUET.exists() and not refresh:
        return load_processed()
    raw = load_raw(nrows=nrows)
    df = prepare(raw)
    save_processed(df)
    return df


def time_split(
    df: pd.DataFrame,
    train_years: tuple[int, ...] = (2014, 2015, 2016),
    val_years: tuple[int, ...] = (2017,),
    test_years: tuple[int, ...] = (2018,),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Vintage-based train / validate / test split. Returns three DataFrames."""
    train = df[df["vintage_year"].isin(train_years)].copy()
    val = df[df["vintage_year"].isin(val_years)].copy()
    test = df[df["vintage_year"].isin(test_years)].copy()
    return train, val, test


if __name__ == "__main__":
    print(f"Reading: {RAW_CSV}")
    df = build_processed(refresh=True)
    print(f"\nProcessed shape: {df.shape}")
    print(f"Saved cache:     {PROCESSED_PARQUET}")
    print(f"\nDefault rate overall: {df['default'].mean() * 100:.2f}%")
    print("\nBy vintage year:")
    print(
        df.groupby("vintage_year")
        .agg(
            n_loans=("default", "size"),
            default_rate_pct=("default", lambda s: round(s.mean() * 100, 2)),
        )
        .to_string()
    )
    train, val, test = time_split(df)
    print(
        f"\nSplit sizes:\n  Train (2014-16): {len(train):>9,}  "
        f"default {train['default'].mean()*100:.2f}%\n"
        f"  Val   (2017)   : {len(val):>9,}  "
        f"default {val['default'].mean()*100:.2f}%\n"
        f"  Test  (2018)   : {len(test):>9,}  "
        f"default {test['default'].mean()*100:.2f}%"
    )
