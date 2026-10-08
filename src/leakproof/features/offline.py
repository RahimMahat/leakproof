"""Point-in-time features, built offline in DuckDB from the shared spec.

Window aggregates use a RANGE frame ending at `1 PRECEDING`, so a transaction never sees itself
or anything from its own second. Label counts merge two event streams per card, label arrivals
and transactions, and take a running sum: a transaction only counts labels that arrived at or
before its own time.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from leakproof.features.spec import LABEL_COUNTS, WINDOW_AGGS, WindowAgg, add_derived

REQUIRED = ["TransactionID", "TransactionDT", "TransactionAmt", "card_id", "isFraud", "label_available_at"]

_SQL = {
    "count": "count(*) over ({w})",
    "sum": "coalesce(sum(amt) over ({w}), 0)",
    "mean": "avg(amt) over ({w})",
    "std": "stddev_samp(amt) over ({w})",
    "min_t": "min(t) over ({w})",
    "max_t": "max(t) over ({w})",
}


def _window_sql(a: WindowAgg) -> str:
    start = "unbounded preceding" if a.window_s is None else f"{a.window_s} preceding"
    frame = f"partition by card_id order by t range between {start} and 1 preceding"
    return f"{_SQL[a.func].format(w=frame)} as {a.name}"


def build_offline_features(df: pd.DataFrame, instant_labels: bool = False) -> pd.DataFrame:
    """One row per TransactionID with every spec feature.

    `instant_labels=True` pretends a label is known one second after its transaction. It exists
    only to measure how much the label delay matters; it is a leak, not a feature.
    """
    src = df[REQUIRED]  # noqa: F841  (referenced by DuckDB below)
    label_time = "TransactionDT + 1" if instant_labels else "label_available_at"
    windows = ",\n            ".join(_window_sql(a) for a in WINDOW_AGGS)
    labeled, fraud = LABEL_COUNTS
    con = duckdb.connect()
    try:
        out = con.sql(
            f"""
            with tx as (
                select TransactionID, card_id, TransactionDT::bigint as t, TransactionAmt::double as amt
                from src
            ),
            win as (
                select TransactionID,
                    {windows}
                from tx
            ),
            events as (  -- kind 0 = a label arrives, kind 1 = a transaction asks "what do I know?"
                select card_id, ({label_time})::double as ev_t, 0 as kind, isFraud::bigint as fraud,
                       null::bigint as TransactionID
                from src
                union all
                select card_id, TransactionDT::double, 1, 0, TransactionID from src
            ),
            running as (
                select TransactionID, kind,
                    sum(1 - kind) over w as {labeled},
                    sum(fraud) over w as {fraud}
                from events
                window w as (partition by card_id order by ev_t, kind
                             rows between unbounded preceding and current row)
            )
            select win.*, r.{labeled}, r.{fraud}
            from win join running r using (TransactionID)
            where r.kind = 1
            order by TransactionID
            """
        ).df()
    finally:
        con.close()
    keys = df[["TransactionID", "TransactionDT", "TransactionAmt"]]
    merged = keys.merge(out, on="TransactionID", how="left", validate="one_to_one")
    return add_derived(merged).drop(columns=["TransactionDT", "TransactionAmt"])
