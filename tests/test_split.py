"""Split and label-delay rules, on synthetic data (no Kaggle download needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakproof.config import DAY
from leakproof.data.split import (
    LEGIT_DELAY_DAYS,
    PERIODS,
    assign_period,
    build_dataset,
    simulate_label_delay,
    summarize,
)


@pytest.fixture
def tx() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 5_000
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n) + 1_000,
            # shuffled on purpose: row order must not matter
            "TransactionDT": rng.permutation(np.sort(rng.integers(DAY, 180 * DAY, n))),
            "isFraud": (rng.random(n) < 0.035).astype(int),
        }
    )


def test_every_row_has_exactly_one_period(tx):
    out = assign_period(tx)
    assert len(out) == len(tx)
    assert set(out["period"]) == set(PERIODS)
    assert out["period"].notna().all()


def test_periods_do_not_overlap_in_time(tx):
    out = assign_period(tx)
    bounds = out.groupby("period")["TransactionDT"].agg(["min", "max"])
    assert bounds.loc["train", "max"] < bounds.loc["valid", "min"]
    assert bounds.loc["valid", "max"] < bounds.loc["holdout", "min"]


def test_period_sizes_are_roughly_70_15_15(tx):
    share = assign_period(tx)["period"].value_counts(normalize=True)
    assert share["train"] == pytest.approx(0.70, abs=0.01)
    assert share["valid"] == pytest.approx(0.15, abs=0.01)
    assert share["holdout"] == pytest.approx(0.15, abs=0.01)


def test_same_timestamp_never_straddles_a_cut():
    df = pd.DataFrame({"TransactionDT": [1] * 7 + [2] * 3, "TransactionID": range(10), "isFraud": 0})
    out = assign_period(df)
    assert out.groupby("TransactionDT")["period"].nunique().max() == 1


def test_label_is_always_available_after_the_transaction(tx):
    out = simulate_label_delay(tx)
    delay_days = (out["label_available_at"] - out["TransactionDT"]) / DAY
    assert (delay_days >= 1).all()
    assert (delay_days[out["isFraud"] == 0] == LEGIT_DELAY_DAYS).all()
    fraud = delay_days[out["isFraud"] == 1]
    assert 10 < fraud.median() < 19 and fraud.max() <= 90


def test_label_delay_is_deterministic_and_order_independent(tx):
    a = simulate_label_delay(tx, seed=7).set_index("TransactionID")["label_available_at"]
    b = simulate_label_delay(tx.sample(frac=1, random_state=1), seed=7).set_index("TransactionID")[
        "label_available_at"
    ]
    pd.testing.assert_series_equal(a.sort_index(), b.sort_index())
    c = simulate_label_delay(tx, seed=8).set_index("TransactionID")["label_available_at"]
    assert not a.sort_index().equals(c.sort_index())


def test_build_dataset_end_to_end(tx, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    tx.assign(TransactionAmt=10.0, card1=1).to_csv(raw / "train_transaction.csv", index=False)
    tx[["TransactionID"]].head(100).assign(DeviceType="mobile").to_csv(
        raw / "train_identity.csv", index=False
    )
    out = build_dataset(raw, tmp_path / "t.parquet")
    df = pd.read_parquet(out)
    assert len(df) == len(tx)  # identity is a left join: no rows lost or duplicated
    assert {"period", "label_available_at", "DeviceType", "TransactionAmt"} <= set(df.columns)
    assert df["TransactionDT"].is_monotonic_increasing
    assert summarize(out)["rows"].sum() == len(tx)
