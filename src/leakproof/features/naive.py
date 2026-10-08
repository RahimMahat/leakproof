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
