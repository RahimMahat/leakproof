"""The careless features, kept on purpose.

Each function makes a mistake that is easy to make and hard to notice, because the offline score
goes *up*. Both compute statistics over every row they are given, with no regard for time:

- `add_global_aggregates`: a card's count and spend statistics include its *future* transactions.
- `add_target_encoding`: a card's fraud rate includes labels that had not arrived yet, and the
  row's own label.

The point-in-time versions live in `features/offline.py`.
"""

from __future__ import annotations

import pandas as pd

GLOBAL_AGG_FEATURES = ["card_tx_count", "card_amt_mean", "card_amt_std", "amt_vs_card_mean"]
TARGET_ENC_FEATURES = ["card_fraud_rate", "card_fraud_count"]


def add_global_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("card_id")["TransactionAmt"]
    mean = g.transform("mean")
    return df.assign(
        card_tx_count=g.transform("count"),
        card_amt_mean=mean,
        card_amt_std=g.transform("std"),
        amt_vs_card_mean=df["TransactionAmt"] / mean,
    )


def add_target_encoding(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("card_id")["isFraud"]
    return df.assign(card_fraud_rate=g.transform("mean"), card_fraud_count=g.transform("sum"))


def naive_serving_features(df: pd.DataFrame) -> pd.DataFrame:
    """The naive features as production can actually serve them.

    Offline, the careless pipeline computed these over every row. At serving time the future
    does not exist: a card's statistics can only cover its transactions so far (this one
    included), and its fraud rate can only use labels that have arrived. `df` needs
    TransactionAmt plus the online aggregates card_cnt_all, card_amt_mean_all, card_amt_std_all,
    card_labeled_cnt and card_fraud_cnt.
    """
    import numpy as np

    amt = df["TransactionAmt"].to_numpy(dtype=float)
    n = df["card_cnt_all"].to_numpy(dtype=float)
    prior_mean = np.nan_to_num(df["card_amt_mean_all"].to_numpy(dtype=float))
    prior_std = np.nan_to_num(df["card_amt_std_all"].to_numpy(dtype=float))

    count = n + 1
    mean = (prior_mean * n + amt) / count
    prior_sumsq = np.maximum(n - 1, 0) * prior_std**2 + n * prior_mean**2
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (prior_sumsq + amt**2 - count * mean**2) / n  # sample variance over count = n + 1 rows
        std = np.where(n >= 1, np.sqrt(np.maximum(var, 0)), np.nan)
        labeled = df["card_labeled_cnt"].to_numpy(dtype=float)
        fraud = df["card_fraud_cnt"].to_numpy(dtype=float)
        rate = np.where(labeled > 0, fraud / labeled, np.nan)
    return df.assign(
        card_tx_count=count,
        card_amt_mean=mean,
        card_amt_std=std,
        amt_vs_card_mean=amt / mean,
        card_fraud_rate=rate,
        card_fraud_count=fraud,
    )
