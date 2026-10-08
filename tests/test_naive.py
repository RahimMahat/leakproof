"""The naive features leak in exactly the ways they are documented to, and the ladder runs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakproof.features.base import BASE_FEATURES, CATEGORICAL, NUMERIC, to_matrix
from leakproof.features.naive import add_global_aggregates, add_target_encoding
from leakproof.model.naive_run import random_split, run_ladder


def test_global_aggregates_see_the_future():
    df = pd.DataFrame({"card_id": ["a", "a", "a", "b"], "TransactionAmt": [10.0, 20.0, 60.0, 5.0]})
    out = add_global_aggregates(df)
    # the first transaction of card "a" already "knows" about the two later ones
    assert out.loc[0, "card_tx_count"] == 3 and out.loc[0, "card_amt_mean"] == 30.0
    assert out.loc[3, "card_tx_count"] == 1 and np.isnan(out.loc[3, "card_amt_std"])


def test_target_encoding_includes_the_rows_own_label():
    df = pd.DataFrame({"card_id": ["a", "a", "b"], "isFraud": [1, 0, 1]})
    out = add_target_encoding(df)
    assert out["card_fraud_rate"].tolist() == [0.5, 0.5, 1.0]  # card "b" is its own label
    assert out["card_fraud_count"].tolist() == [1, 1, 1]


def test_random_split_is_a_partition():
    tr, st, te = random_split(1000)
    assert len(tr) == 700 and len(st) == 150 and len(te) == 150
    assert len(set(tr) | set(st) | set(te)) == 1000
    assert (random_split(1000)[2] == te).all()  # seeded


def _synthetic(n: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUMERIC})
    for c in CATEGORICAL:
        df[c] = rng.choice(["x", "y", None], size=n)
    df["TransactionAmt"] = rng.gamma(2.0, 50.0, n)
    df["card_id"] = rng.integers(0, 400, n).astype(str)
    # fraud is a property of the card, so card-level label statistics leak it
    bad_cards = set(rng.choice(400, 30, replace=False).astype(str))
    df["isFraud"] = (df["card_id"].isin(bad_cards) & (rng.random(n) < 0.6)).astype(int)
    df["TransactionDT"] = np.sort(rng.integers(0, 10**7, n))
    df["period"] = np.where(np.arange(n) < n * 0.82, "train", "valid")
    return df


def test_matrix_has_base_features_and_categoricals():
    X = to_matrix(_synthetic(200))
    assert list(X.columns) == BASE_FEATURES
    assert all(str(X[c].dtype) == "category" for c in CATEGORICAL)


def test_ladder_runs_and_target_encoding_inflates_the_score():
    results = {r["variant"]: r for r in run_ladder(_synthetic())}
    assert list(results) == ["time_split_base", "random_base", "random_global_aggs", "random_target_enc"]
    assert results["random_target_enc"]["roc_auc"] > results["random_base"]["roc_auc"] + 0.1
    assert results["random_target_enc"]["roc_auc"] == pytest.approx(1.0, abs=0.05)
