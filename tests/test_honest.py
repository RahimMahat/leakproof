"""Label maturity and the honest ladder, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from leakproof.config import DAY
from leakproof.model.honest_run import mature_positions, run_ladder, time_ordered_split
from tests.test_naive import _synthetic


def test_only_rows_with_arrived_labels_are_mature():
    df = pd.DataFrame(
        {
            "period": ["train", "train", "train", "valid"],
            "label_available_at": [5.0, 10.0, 11.0, 1.0],
        }
    )
    assert mature_positions(df, "train", cutoff=10.0).tolist() == [0, 1]  # 11 has not arrived; valid excluded


def test_early_stopping_rows_come_after_training_rows():
    tr, st = time_ordered_split(np.arange(100))
    assert tr.max() < st.min() and len(st) == 15


def test_honest_ladder_runs_and_trains_on_fewer_rows():
    df = _synthetic()
    # legit labels take 30 days, fraud labels 10: recent train rows are immature at the cutoff
    df["TransactionDT"] = np.linspace(0, 120 * DAY, len(df)).astype(int)
    df["TransactionID"] = np.arange(len(df))
    df["label_available_at"] = df["TransactionDT"] + np.where(df["isFraud"] == 1, 10, 30) * DAY
    results = {r["variant"]: r for r in run_ladder(df)}
    assert list(results) == ["base_mature", "pit_instant_labels", "pit_honest"]
    assert results["pit_honest"]["train_rows"] < results["pit_instant_labels"]["train_rows"]
    assert results["pit_honest"]["train_rows"] == results["base_mature"]["train_rows"]
