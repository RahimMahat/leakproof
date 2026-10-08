"""Columns every model sees, the card identifier, and loading into pandas.

`card_id` approximates one account: the card fields plus the day the card was first seen
(transaction day minus D1, "days since first transaction"). Everything in it is known at
transaction time, so using it is not leakage.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from leakproof.config import DATASET_PATH, DAY

KEYS = ["TransactionID", "TransactionDT", "isFraud", "period", "label_available_at"]
NUMERIC = [
    "TransactionAmt", "dist1", "dist2", "card1", "card2", "card3", "card5", "addr1", "addr2",
    *[f"C{i}" for i in range(1, 15)],
    *[f"D{i}" for i in range(1, 16)],
]  # fmt: skip
CATEGORICAL = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType",
    *[f"M{i}" for i in range(1, 10)],
]  # fmt: skip
BASE_FEATURES = [*NUMERIC, *CATEGORICAL]

CARD_ID_SQL = (
    "concat_ws('|', coalesce(card1::varchar, ''), coalesce(card2::varchar, ''), "
    "coalesce(card3::varchar, ''), coalesce(card5::varchar, ''), coalesce(addr1::varchar, ''), "
    f"coalesce(cast(floor(TransactionDT / {DAY} - D1) as int)::varchar, ''))"
)


def load_frame(periods: Iterable[str], path: Path = DATASET_PATH) -> pd.DataFrame:
    """Keys, base features and `card_id` for the given periods, ordered by time."""
    wanted = ", ".join(f"'{p}'" for p in periods)
    cols = ", ".join(f'"{c}"' for c in [*KEYS, *BASE_FEATURES])
    con = duckdb.connect()
    try:
        df = con.sql(
            f"select {cols}, {CARD_ID_SQL} as card_id from read_parquet('{path.as_posix()}') "
            f"where period in ({wanted}) order by TransactionDT, TransactionID"
        ).df()
    finally:
        con.close()
    return df


def to_matrix(df: pd.DataFrame, extra: Iterable[str] = ()) -> pd.DataFrame:
    """Model input: base features plus `extra`, with categoricals typed for LightGBM."""
    X = df[[*BASE_FEATURES, *extra]].copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype("category")
    return X


def card_id_from_row(row: dict[str, Any]) -> str:
    """`card_id` for one transaction, identical to CARD_ID_SQL (a test checks this on real data)."""

    def num(key: str, as_int: bool = False) -> str:
        v = row.get(key)
        if v is None or v != v:  # None or NaN
            return ""
        return str(int(v)) if as_int else repr(float(v))

    d1, dt = row.get("D1"), row.get("TransactionDT")
    if d1 is None or d1 != d1 or dt is None:
        return "|".join([num("card1", True), num("card2"), num("card3"), num("card5"), num("addr1"), ""])
    first_seen = str(math.floor(float(dt) / DAY - float(d1)))
    return "|".join([num("card1", True), num("card2"), num("card3"), num("card5"), num("addr1"), first_seen])
