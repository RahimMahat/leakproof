"""The event stream: what gets produced, and what the consumer does with each event.

Two kinds of event share one ordered stream:

    {"kind": "label", "t": ..., "card_id": ..., "is_fraud": 0|1}
    {"kind": "tx",    "t": ..., "card_id": ..., "txid": ..., "amt": ...}

They are ordered by time, and at equal time a label comes first, which is the spec's rule that a
transaction may use labels with label_available_at <= t.

Why one stream: with separate topics, the order between a label and a transaction is not
guaranteed without event-time watermarks. One single-partition stream makes the order a property
of the data, so the online features are reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np
import pandas as pd

from leakproof.features.online import OnlineFeatureStore

Event = dict[str, Any]
BATCH = 5_000


def build_events(df: pd.DataFrame, start: float = -np.inf, end: float = np.inf) -> list[Event]:
    """Events with start <= t <= end, in stream order. `df` needs the keys, amount and card_id."""
    tx = df[(df["TransactionDT"] >= start) & (df["TransactionDT"] <= end)]
    lab = df[(df["label_available_at"] >= start) & (df["label_available_at"] <= end)]
    events: list[tuple[float, int, int, Event]] = []
    for r in lab.itertuples():
        t = float(r.label_available_at)
        ev = {"kind": "label", "t": t, "card_id": r.card_id, "is_fraud": int(r.isFraud)}
        events.append((t, 0, int(r.TransactionID), ev))
    for r in tx.itertuples():
        t = int(r.TransactionDT)
        ev = {
            "kind": "tx",
            "t": t,
            "card_id": r.card_id,
            "txid": int(r.TransactionID),
            "amt": float(r.TransactionAmt),
        }
        events.append((float(t), 1, int(r.TransactionID), ev))
    events.sort(key=lambda e: e[:3])
    return [e[3] for e in events]


def process_event(store: OnlineFeatureStore, ev: Event) -> dict[str, float] | None:
    """Apply one event. For a transaction, return its features, read *before* it is recorded."""
    if ev["kind"] == "label":
        store.apply_label(ev["card_id"], ev["is_fraud"])
        return None
    features = store.raw_features(ev["card_id"], ev["t"])
    store.apply_transaction(ev["card_id"], ev["txid"], ev["t"], ev["amt"])
    return {"TransactionID": ev["txid"], **features}


def process_stream(store: OnlineFeatureStore, events: Iterable[Event]) -> Iterator[dict[str, float]]:
    for ev in events:
        if (row := process_event(store, ev)) is not None:
            yield row


def backfill(store: OnlineFeatureStore, events: Iterable[Event]) -> int:
    """Load history into the store without computing features (bulk writes). Returns events applied."""
    n = 0
    pipe = store.r.pipeline(transaction=False)
    for ev in events:
        if ev["kind"] == "label":
            store.apply_label(ev["card_id"], ev["is_fraud"], to=pipe)
        else:
            store.apply_transaction(ev["card_id"], ev["txid"], ev["t"], ev["amt"], to=pipe)
        n += 1
        if n % BATCH == 0:
            pipe.execute()
    pipe.execute()
    return n
