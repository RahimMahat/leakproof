"""The leak-free pipeline, and the two shortcuts that are tempting on the way to it.

All three variants split by time and are tested on the valid period. The holdout period is not
loaded at all; it stays untouched until the live replay.

    base_mature         base features; trained only on rows whose label had arrived by the cutoff
    pit_instant_labels  + card history, but labels are assumed known one second after the
                        transaction, and every train row is used (two label-timing leaks)
    pit_honest          + card history with delayed labels; trained only on mature rows

"Mature" means label_available_at <= training cutoff. At the moment a model is trained, recent
transactions have no label yet: a chargeback may still arrive. Training on them as if they were
confirmed uses information from the future.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leakproof.config import PROJECT_ROOT
from leakproof.features.base import load_frame
from leakproof.features.offline import build_offline_features
from leakproof.features.spec import LABEL_FEATURES, PIT_FEATURES
from leakproof.model.naive_run import STOP_FRAC, run_variant

RESULTS_PATH = PROJECT_ROOT / "results" / "honest.json"


def time_ordered_split(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train and early-stopping positions: the last STOP_FRAC of the (time-ordered) rows stop."""
    cut = int(len(positions) * (1 - STOP_FRAC))
    return positions[:cut], positions[cut:]


def mature_positions(df: pd.DataFrame, period: str, cutoff: float) -> np.ndarray:
    """Rows of `period` whose label had arrived by `cutoff`."""
    return np.flatnonzero((df["period"] == period) & (df["label_available_at"] <= cutoff))


def with_features(df: pd.DataFrame, instant_labels: bool = False) -> pd.DataFrame:
    feats = build_offline_features(df, instant_labels=instant_labels)
    return df.merge(feats, on="TransactionID", how="left", validate="one_to_one")


def run_ladder(df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    # History features are built over the development data only, in time order.
    df = load_frame(("train", "valid")) if df is None else df
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    test = np.flatnonzero(df["period"] == "valid")
    cutoff = float(df.loc[test, "TransactionDT"].min())  # the model is trained when valid begins
    extra = [*PIT_FEATURES, *LABEL_FEATURES]

    all_train = np.flatnonzero(df["period"] == "train")
    mature = mature_positions(df, "train", cutoff)
    results: list[dict[str, Any]] = []

    def record(variant: str, frame: pd.DataFrame, cols: list[str], rows: np.ndarray) -> None:
        tr, st = time_ordered_split(rows)
        results.append(
            {
                "variant": variant,
                "split": "time",
                **run_variant(frame, cols, tr, st, test),
                "train_rows": len(rows),
            }
        )

    record("base_mature", df, [], mature)
    record("pit_instant_labels", with_features(df, instant_labels=True), extra, all_train)
    record("pit_honest", with_features(df), extra, mature)
    return results
