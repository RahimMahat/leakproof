"""Replay the start of the holdout period through the HTTP scoring service.

Measures latency over real HTTP and Redis, and checks that the service's scores equal the batch
scores for the same transactions (same model, same features, so they must).
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from leakproof.config import PROJECT_ROOT

RESULTS_PATH = PROJECT_ROOT / "results" / "serving.json"
PORT = 8077


def _percentiles(values: list[float]) -> dict[str, float]:
    a = np.array(values)
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in (50, 95, 99)}


def run_replay(limit: int = 2000) -> dict[str, Any]:
    import httpx
    import joblib
    import redis
    import uvicorn

    from leakproof.config import REDIS_URL
    from leakproof.features.base import BASE_FEATURES, KEYS, load_frame, to_matrix
    from leakproof.features.online import OnlineFeatureStore
    from leakproof.model.final import MODEL_DIR
    from leakproof.serve.app import create_app
    from leakproof.stream.events import backfill, build_events
    from leakproof.stream.parity import ONLINE_FEATURES_PATH

    df = load_frame(("train", "valid", "holdout"))
    holdout = df[df["period"] == "holdout"]
    start = float(holdout["TransactionDT"].min())

    client = redis.Redis.from_url(REDIS_URL)
    client.flushdb()
    store = OnlineFeatureStore(client)
    backfill(store, build_events(df, end=np.nextafter(start, -np.inf)))

    bundle = joblib.load(MODEL_DIR / "honest.joblib")
    server = uvicorn.Server(
        uvicorn.Config(create_app(store, bundle["model"], bundle["extra"]), port=PORT, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)

    # The first `limit` holdout transactions, with the label events that arrive between them.
    sample = holdout.head(limit)
    end = float(sample["TransactionDT"].max())
    events = [e for e in build_events(df, start=start, end=end)]
    wanted = set(sample["TransactionID"])
    payloads = {
        int(r["TransactionID"]): {k: (None if v != v else v) for k, v in r.items()}
        for r in sample[[*KEYS[:2], *BASE_FEATURES]].to_dict("records")
    }

    server_ms, client_ms, scores = [], [], {}
    t_run = time.perf_counter()
    with httpx.Client(base_url=f"http://127.0.0.1:{PORT}", timeout=30) as http:
        for ev in events:
            if ev["kind"] == "label":
                store.apply_label(ev["card_id"], ev["is_fraud"])
            elif ev["txid"] in wanted:
                t0 = time.perf_counter()
                body = http.post("/score", json=payloads[ev["txid"]]).raise_for_status().json()
                client_ms.append((time.perf_counter() - t0) * 1000)
                server_ms.append(body["latency_ms"])
                scores[ev["txid"]] = body["score"]
            else:  # a same-second transaction beyond the sample: keep the store consistent
                store.apply_transaction(ev["card_id"], ev["txid"], ev["t"], ev["amt"])
    elapsed = time.perf_counter() - t_run
    server.should_exit = True

    # Batch scores for the same rows, from the stream features saved by the parity run.
    online = pd.read_parquet(ONLINE_FEATURES_PATH)
    batch_frame = sample.merge(online, on="TransactionID", validate="one_to_one")
    batch = np.asarray(bundle["model"].predict_proba(to_matrix(batch_frame, bundle["extra"])))[:, 1]
    api = np.array([scores[i] for i in batch_frame["TransactionID"]])
    return {
        "requests": len(scores),
        "requests_per_s": round(len(scores) / elapsed, 1),
        "server_latency_ms": _percentiles(server_ms),
        "client_latency_ms": _percentiles(client_ms),
        "max_score_diff_vs_batch": float(np.abs(api - batch).max()),
    }
