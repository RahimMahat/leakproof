"""The feature spec: declared once, implemented twice (offline in DuckDB, online in Redis).

The time rule, shared by both implementations:

    A feature for a transaction at time t may use
      - the same card's transactions with TransactionDT <= t - 1   (earlier seconds only)
      - labels with label_available_at <= t                        (labels that have arrived)

Same-second transactions are excluded on purpose: the stream gives no reliable order within a
second, so neither implementation is allowed to depend on one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

HOUR, DAY, WEEK = 3_600, 86_400, 604_800


@dataclass(frozen=True)
class WindowAgg:
    """An aggregate over a card's earlier transactions. `window_s=None` means all history."""

    name: str
    func: str  # count | sum | mean | std | min_t | max_t
    window_s: int | None = None


WINDOW_AGGS: tuple[WindowAgg, ...] = (
    WindowAgg("card_cnt_1h", "count", HOUR),
    WindowAgg("card_cnt_24h", "count", DAY),
    WindowAgg("card_cnt_7d", "count", WEEK),
    WindowAgg("card_amt_sum_24h", "sum", DAY),
    WindowAgg("card_amt_sum_7d", "sum", WEEK),
    WindowAgg("card_cnt_all", "count"),
    WindowAgg("card_amt_mean_all", "mean"),
    WindowAgg("card_amt_std_all", "std"),
    WindowAgg("card_first_t", "min_t"),
    WindowAgg("card_last_t", "max_t"),
)

# Counts of the card's earlier transactions whose label has arrived, and how many were fraud.
LABEL_COUNTS = ("card_labeled_cnt", "card_fraud_cnt")

# What the model sees. The raw first/last timestamps are turned into durations by add_derived.
PIT_FEATURES = [
    "card_cnt_1h",
    "card_cnt_24h",
    "card_cnt_7d",
    "card_amt_sum_24h",
    "card_amt_sum_7d",
    "card_cnt_all",
    "card_amt_mean_all",
    "card_amt_std_all",
    "secs_since_last_tx",
    "card_age_s",
    "amt_vs_card_mean",
]
LABEL_FEATURES = ["card_labeled_cnt", "card_fraud_cnt", "card_fraud_rate"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Features computed from the raw aggregates. Used by both the offline and the online path,
    so the arithmetic can't drift between them. Needs TransactionDT and TransactionAmt."""
    return df.assign(
        secs_since_last_tx=df["TransactionDT"] - df["card_last_t"],
        card_age_s=df["TransactionDT"] - df["card_first_t"],
        amt_vs_card_mean=df["TransactionAmt"] / df["card_amt_mean_all"],
        card_fraud_rate=df["card_fraud_cnt"] / df["card_labeled_cnt"].where(df["card_labeled_cnt"] > 0),
    )
