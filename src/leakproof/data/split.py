"""Time-based split and label-delay simulation.

Two rules that the rest of the project depends on:

1. **Periods are cut on time, never on shuffled rows.** train is the earliest 70% of transactions,
   valid the next 15%, holdout the last 15%. The cut points are TransactionDT values, so two
   transactions at the same second always land in the same period.
2. **A label is not known when the transaction happens.** IEEE-CIS has no chargeback dates, so the
   delay is simulated: fraud is confirmed after a lognormal delay (median 14 days), and a
   legitimate transaction is assumed fine once 30 days pass with no chargeback.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from leakproof.config import DATASET_PATH, DAY, RAW_DIR, SEED

PERIODS = ("train", "valid", "holdout")
TRAIN_FRAC, VALID_FRAC = 0.70, 0.15

FRAUD_DELAY_MEDIAN_DAYS = 14.0
FRAUD_DELAY_SIGMA = 0.6
FRAUD_DELAY_RANGE_DAYS = (1.0, 90.0)
LEGIT_DELAY_DAYS = 30.0


def period_cutoffs(dt: pd.Series) -> tuple[float, float]:
    """TransactionDT values where valid and holdout begin."""
    ordered = np.sort(dt.to_numpy())
    n = len(ordered)
    return float(ordered[int(n * TRAIN_FRAC)]), float(ordered[int(n * (TRAIN_FRAC + VALID_FRAC))])


def assign_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `period` column (train / valid / holdout) from TransactionDT."""
    valid_from, holdout_from = period_cutoffs(df["TransactionDT"])
    dt = df["TransactionDT"]
    period = np.where(dt < valid_from, "train", np.where(dt < holdout_from, "valid", "holdout"))
    return df.assign(period=period)


def simulate_label_delay(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Add `label_available_at` (seconds, same clock as TransactionDT). Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    # Draw for every row in TransactionID order so the result doesn't depend on row order.
    order = np.argsort(df["TransactionID"].to_numpy(), kind="stable")
    draws = np.empty(len(df))
    draws[order] = rng.lognormal(np.log(FRAUD_DELAY_MEDIAN_DAYS), FRAUD_DELAY_SIGMA, len(df))
    fraud_delay = np.clip(draws, *FRAUD_DELAY_RANGE_DAYS)
    delay_days = np.where(df["isFraud"].to_numpy() == 1, fraud_delay, LEGIT_DELAY_DAYS)
    return df.assign(label_available_at=df["TransactionDT"].to_numpy() + delay_days * DAY)


def build_dataset(raw_dir: Path = RAW_DIR, out_path: Path = DATASET_PATH, seed: int = SEED) -> Path:
    """Join the raw CSVs, add `period` and `label_available_at`, write one parquet file.

    DuckDB streams the 680MB CSV; only three narrow columns are ever held in pandas.
    """
    tx, ident = raw_dir / "train_transaction.csv", raw_dir / "train_identity.csv"
    for f in (tx, ident):
        if not f.exists():
            raise FileNotFoundError(f"{f} missing; run `leakproof data download`")
    con = duckdb.connect()
    try:
        keys = con.sql(f"select TransactionID, TransactionDT, isFraud from read_csv('{tx.as_posix()}')").df()
        keys = simulate_label_delay(assign_period(keys), seed)[
            ["TransactionID", "period", "label_available_at"]
        ]
        con.register("keys", keys)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        con.sql(
            f"""
            copy (
                select t.*, i.* exclude (TransactionID), k.period, k.label_available_at
                from read_csv('{tx.as_posix()}') t
                join keys k using (TransactionID)
                left join read_csv('{ident.as_posix()}') i using (TransactionID)
                order by t.TransactionDT, t.TransactionID
            ) to '{out_path.as_posix()}' (format parquet)
            """
        )
    finally:
        con.close()
    return out_path


def summarize(path: Path = DATASET_PATH) -> pd.DataFrame:
    """Rows, time range and fraud rate per period."""
    con = duckdb.connect()
    try:
        return con.sql(
            f"""
            select period,
                   count(*) as rows,
                   round(min(TransactionDT) / {DAY}, 1) as from_day,
                   round(max(TransactionDT) / {DAY}, 1) as to_day,
                   sum(isFraud) as frauds,
                   round(100.0 * avg(isFraud), 2) as fraud_pct
            from read_parquet('{path.as_posix()}')
            group by period order by from_day
            """
        ).df()
    finally:
        con.close()
