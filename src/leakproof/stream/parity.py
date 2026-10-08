"""Training-serving parity: do the online features equal the offline ones?

`run_parity` exercises the real path on the holdout period:

    history (train + valid) --backfill--> Redis
    holdout events --produce--> Redpanda --consume--> feature consumer --> online features
    compare with the offline features for the same transactions
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leakproof.config import DATA_DIR, PROJECT_ROOT
from leakproof.features.spec import LABEL_FEATURES, PIT_FEATURES

FEATURES = [*PIT_FEATURES, *LABEL_FEATURES]
RESULTS_PATH = PROJECT_ROOT / "results" / "parity.json"
ONLINE_FEATURES_PATH = DATA_DIR / "online_features_holdout.parquet"
RTOL, ATOL = 1e-6, 1e-6  # floating-point noise only; a real skew is far larger


def compare_features(online: pd.DataFrame, offline: pd.DataFrame) -> dict[str, Any]:
    """Row and per-feature mismatch counts between two feature frames keyed by TransactionID."""
    a = online.set_index("TransactionID").sort_index()[FEATURES].astype(float)
    b = offline.set_index("TransactionID").sort_index()[FEATURES].astype(float)
    if not a.index.equals(b.index):
        raise ValueError(f"different transactions: {len(a)} online vs {len(b)} offline")
    close = np.isclose(a.to_numpy(), b.to_numpy(), rtol=RTOL, atol=ATOL, equal_nan=True)
    diff = np.abs(a.to_numpy() - b.to_numpy())
    by_feature = {
        f: {
            "mismatches": int((~close[:, i]).sum()),
            "max_abs_diff": float(np.nanmax(diff[:, i], initial=0.0)),
        }
        for i, f in enumerate(FEATURES)
    }
    bad_rows = int((~close.all(axis=1)).sum())
    return {
        "rows": len(a),
        "mismatched_rows": bad_rows,
        "mismatch_rate": bad_rows / len(a) if len(a) else 0.0,
        "by_feature": by_feature,
    }


def run_parity(limit: int | None = None) -> dict[str, Any]:
    """Backfill history, stream the holdout period through Redpanda, compare with offline."""
    import redis

    from leakproof.config import REDIS_URL
    from leakproof.features.base import load_frame
    from leakproof.features.offline import build_offline_features
    from leakproof.features.online import OnlineFeatureStore
    from leakproof.features.spec import add_derived
    from leakproof.stream import kafka_io
    from leakproof.stream.events import backfill, build_events, process_stream

    df = load_frame(("train", "valid", "holdout"))
    holdout = df[df["period"] == "holdout"]
    start = float(holdout["TransactionDT"].min())
    timings: dict[str, float] = {}

    client = redis.Redis.from_url(REDIS_URL)
    client.flushdb()
    store = OnlineFeatureStore(client)

    t0 = time.perf_counter()
    history = build_events(df, end=np.nextafter(start, -np.inf))
    n_history = backfill(store, history)
    timings["backfill_s"] = round(time.perf_counter() - t0, 1)

    live = build_events(df, start=start)
    if limit:
        live = live[:limit]
    t0 = time.perf_counter()
    topic = kafka_io.fresh_topic("events")
    kafka_io.produce(topic, live)
    timings["produce_s"] = round(time.perf_counter() - t0, 1)

    t0 = time.perf_counter()
    rows = list(process_stream(store, kafka_io.consume(topic)))
    timings["consume_s"] = round(time.perf_counter() - t0, 1)
    kafka_io.delete_topic(topic)

    keys = holdout[["TransactionID", "TransactionDT", "TransactionAmt"]]
    online = add_derived(keys.merge(pd.DataFrame(rows), on="TransactionID", validate="one_to_one"))
    online = online.drop(columns=["TransactionDT", "TransactionAmt"])
    online.to_parquet(ONLINE_FEATURES_PATH, index=False)

    offline = build_offline_features(df)
    offline = offline[offline["TransactionID"].isin(online["TransactionID"])]
    report = compare_features(online, offline)
    report.update(
        history_events=n_history,
        live_events=len(live),
        events_per_s=round(len(live) / timings["consume_s"]) if timings["consume_s"] else None,
        timings=timings,
    )
    return report


def save(report: dict[str, Any], path: Path = RESULTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path
