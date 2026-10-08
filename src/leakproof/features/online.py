"""Point-in-time features, served online from Redis using the shared spec.

State per card:
    tx:{card_id}    sorted set, score = TransactionDT, member = "{TransactionID}:{amount}"
    lab:{card_id}   hash with `labeled` and `fraud` counters, incremented when a label arrives

A read at time t takes the card's transactions with score <= t - 1 (the spec's time rule) and
computes every window aggregate from them in one pass. Label counters need no time filter:
labels are applied in arrival order, so the counters already mean "arrived by now".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from leakproof.features.spec import LABEL_COUNTS, WINDOW_AGGS

NAN = float("nan")


def _aggregate(func: str, times: np.ndarray, amts: np.ndarray) -> float:
    n = len(times)
    if func == "count":
        return float(n)
    if func == "sum":
        return float(amts.sum()) if n else 0.0
    if func == "mean":
        return float(amts.mean()) if n else NAN
    if func == "std":
        return float(amts.std(ddof=1)) if n > 1 else NAN
    if func == "min_t":
        return float(times.min()) if n else NAN
    if func == "max_t":
        return float(times.max()) if n else NAN
    raise ValueError(f"unknown aggregate {func}")


class OnlineFeatureStore:
    def __init__(self, client: Any) -> None:
        self.r = client

    # -- writes (pass a pipeline as `to` to batch them) --------------------------------------
    def apply_transaction(self, card_id: str, txid: int, t: int, amt: float, to: Any = None) -> None:
        # ponytail: a card's full history lives in one sorted set and is read whole. Fine here
        # (the longest card has ~1.4k transactions). For long-lived cards, trim to the largest
        # window and keep running count/sum/sum-of-squares for the all-history aggregates.
        (to or self.r).zadd(f"tx:{card_id}", {f"{txid}:{amt!r}": t})

    def apply_label(self, card_id: str, is_fraud: int, to: Any = None) -> None:
        target = to or self.r
        target.hincrby(f"lab:{card_id}", "labeled", 1)
        if is_fraud:
            target.hincrby(f"lab:{card_id}", "fraud", 1)

    # -- read ---------------------------------------------------------------------------------
    def raw_features(self, card_id: str, t: int) -> dict[str, float]:
        """Every spec aggregate for a transaction of `card_id` at time `t`, before it is applied."""
        pipe = self.r.pipeline()
        pipe.zrangebyscore(f"tx:{card_id}", "-inf", t - 1, withscores=True)
        pipe.hmget(f"lab:{card_id}", "labeled", "fraud")
        history, (labeled, fraud) = pipe.execute()
        times = np.array([score for _, score in history], dtype=float)
        amts = np.array([float(_text(member).split(":", 1)[1]) for member, _ in history], dtype=float)

        out: dict[str, float] = {}
        for a in WINDOW_AGGS:
            keep = slice(None) if a.window_s is None else times >= t - a.window_s
            out[a.name] = _aggregate(a.func, times[keep], amts[keep])
        out[LABEL_COUNTS[0]] = float(labeled or 0)
        out[LABEL_COUNTS[1]] = float(fraud or 0)
        return out


def _text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)
