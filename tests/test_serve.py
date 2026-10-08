"""Scoring service and the fast single-row scorer."""

from __future__ import annotations

import fakeredis
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from leakproof.config import DATASET_PATH, DAY
from leakproof.features.base import BASE_FEATURES, card_id_from_row, load_frame, to_matrix
from leakproof.features.online import OnlineFeatureStore
from leakproof.model.final import HONEST_EXTRA, train_honest
from leakproof.model.honest_run import with_features
from leakproof.serve.app import create_app
from leakproof.serve.scorer import FastScorer
from tests.test_naive import _synthetic


@pytest.fixture(scope="module")
def dev() -> pd.DataFrame:
    df = _synthetic(4000)
    df["TransactionDT"] = np.linspace(DAY, 100 * DAY, len(df)).astype(int)
    df["TransactionID"] = np.arange(len(df))
    df["label_available_at"] = df["TransactionDT"] + np.where(df["isFraud"] == 1, 10, 30) * DAY
    return df


@pytest.fixture(scope="module")
def model(dev):
    return train_honest(dev, cutoff=float(dev["TransactionDT"].max()))[0]


def _records(frame: pd.DataFrame) -> list[dict]:
    return [{k: (None if v != v else v) for k, v in r.items()} for r in frame.to_dict("records")]


def test_fast_scorer_equals_the_pandas_path(dev, model):
    frame = with_features(dev).tail(400)
    slow = np.asarray(model.predict_proba(to_matrix(frame, HONEST_EXTRA)))[:, 1]
    scorer = FastScorer(model, HONEST_EXTRA)
    fast = np.array([scorer.score(r) for r in _records(frame)])
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-12)


def test_fast_scorer_treats_unseen_and_missing_categories_as_missing(dev, model):
    scorer = FastScorer(model, HONEST_EXTRA)
    row = _records(with_features(dev).tail(1))[0]
    unseen = scorer.score({**row, "ProductCD": "never-seen"})
    assert unseen == scorer.score({**row, "ProductCD": None})
    assert 0.0 <= unseen <= 1.0


def test_score_endpoint_reads_then_records(dev, model):
    store = OnlineFeatureStore(fakeredis.FakeRedis())
    tx = {
        k: v
        for k, v in _records(dev.tail(1))[0].items()
        if k in [*BASE_FEATURES, "TransactionID", "TransactionDT"]
    }
    with TestClient(create_app(store, model, HONEST_EXTRA)) as client:
        first = client.post("/score", json=tx)
        assert first.status_code == 200, first.text
        body = first.json()
        assert 0.0 <= body["score"] <= 1.0 and body["card_cnt_24h"] == 0  # it did not see itself
        later = {**tx, "TransactionID": tx["TransactionID"] + 1, "TransactionDT": tx["TransactionDT"] + 60}
        assert client.post("/score", json=later).json()["card_cnt_24h"] == 1  # the first one was recorded
        assert client.post("/score", json={"TransactionID": 1, "TransactionDT": 5}).status_code == 422
        assert client.get("/health").json() == {"status": "ok"}


def test_python_card_id_matches_sql_on_real_data():
    if not DATASET_PATH.exists():
        pytest.skip("dataset not built; run `leakproof data build`")
    df = load_frame(("valid",)).head(20_000)
    got = [card_id_from_row(r) for r in df.to_dict("records")]
    assert got == df["card_id"].tolist()
