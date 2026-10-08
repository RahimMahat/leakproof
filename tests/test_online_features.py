"""Online features equal offline features (training-serving parity), using an in-memory Redis."""

from __future__ import annotations

import fakeredis
import pandas as pd

from leakproof.config import DAY
from leakproof.features.offline import build_offline_features
from leakproof.features.online import OnlineFeatureStore
from leakproof.features.spec import add_derived
from leakproof.stream.events import backfill, build_events, process_stream
from leakproof.stream.parity import compare_features
from tests.test_offline_features import synthetic


def store() -> OnlineFeatureStore:
    return OnlineFeatureStore(fakeredis.FakeRedis())


def online(df: pd.DataFrame, rows: list[dict[str, float]]) -> pd.DataFrame:
    keys = df[["TransactionID", "TransactionDT", "TransactionAmt"]]
    merged = keys.merge(pd.DataFrame(rows), on="TransactionID", validate="one_to_one")
    return add_derived(merged).drop(columns=["TransactionDT", "TransactionAmt"])


def test_events_are_ordered_and_labels_come_first_at_equal_time():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [100, 500],
            "TransactionAmt": [10.0, 20.0],
            "card_id": ["a", "a"],
            "isFraud": [1, 0],
            "label_available_at": [500.0, 900.0],
        }
    )
    kinds = [(e["kind"], e["t"]) for e in build_events(df)]
    assert kinds == [("tx", 100), ("label", 500.0), ("tx", 500), ("label", 900.0)]


def test_streaming_everything_matches_offline():
    df = synthetic()
    rows = list(process_stream(store(), build_events(df)))
    report = compare_features(online(df, rows), build_offline_features(df))
    assert report["rows"] == len(df)
    assert report["mismatched_rows"] == 0, report


def test_backfill_then_stream_matches_offline_on_the_streamed_part():
    df = synthetic()
    cut = 25 * DAY
    s = store()
    backfill(s, build_events(df, end=cut))
    rows = list(process_stream(s, build_events(df, start=cut + 1e-9)))
    late = df[df["TransactionDT"] > cut]
    assert len(rows) == len(late)
    offline = build_offline_features(df)
    report = compare_features(
        online(late, rows), offline[offline["TransactionID"].isin(late["TransactionID"])]
    )
    assert report["mismatched_rows"] == 0, report


def test_parity_check_detects_skew():
    df = synthetic(300)
    offline = build_offline_features(df)
    skewed = offline.copy()
    skewed.loc[skewed.index[:7], "card_cnt_24h"] += 1
    report = compare_features(skewed, offline)
    assert report["mismatched_rows"] == 7 and report["by_feature"]["card_cnt_24h"]["mismatches"] == 7
