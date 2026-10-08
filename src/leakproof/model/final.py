"""The headline: each model's offline score next to its score on live features.

Both models are trained as of the moment the holdout period begins, on development data only
(the train and valid periods), then scored on the holdout period with the features the stream
produced for it.

    naive    random split of the development data, card aggregates and fraud rate over all rows
    honest   point-in-time card history, delayed labels, trained only on mature labels

"Live" for the naive model means its features served the only way production can serve them:
from the card's history so far and from labels that have arrived (`naive_serving_features`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from leakproof.config import DATA_DIR, PROJECT_ROOT
from leakproof.features.base import load_frame, to_matrix
from leakproof.features.naive import (
    GLOBAL_AGG_FEATURES,
    TARGET_ENC_FEATURES,
    add_global_aggregates,
    add_target_encoding,
    naive_serving_features,
)
from leakproof.features.spec import LABEL_FEATURES, PIT_FEATURES
from leakproof.model.honest_run import time_ordered_split, with_features
from leakproof.model.naive_run import random_split
from leakproof.model.train import fit, score
from leakproof.stream.parity import ONLINE_FEATURES_PATH

MODEL_DIR = DATA_DIR / "models"
RESULTS_PATH = PROJECT_ROOT / "results" / "headline.json"
NAIVE_EXTRA = [*GLOBAL_AGG_FEATURES, *TARGET_ENC_FEATURES]
HONEST_EXTRA = [*PIT_FEATURES, *LABEL_FEATURES]


def train_naive(dev: pd.DataFrame) -> tuple[Any, dict[str, float]]:
    """The fully careless model and the offline score it reports for itself."""
    df = add_target_encoding(add_global_aggregates(dev))
    X, y = to_matrix(df, NAIVE_EXTRA), df["isFraud"]
    tr, st, te = random_split(len(df))
    model = fit(X.iloc[tr], y.iloc[tr], X.iloc[st], y.iloc[st])
    return model, score(model, X.iloc[te], y.iloc[te])


def train_honest(dev: pd.DataFrame, cutoff: float) -> tuple[Any, int]:
    """Point-in-time features, trained on rows whose label had arrived by `cutoff`."""
    df = with_features(dev)
    mature = np.flatnonzero(df["label_available_at"] <= cutoff)
    tr, st = time_ordered_split(mature)
    X, y = to_matrix(df, HONEST_EXTRA), df["isFraud"]
    return fit(X.iloc[tr], y.iloc[tr], X.iloc[st], y.iloc[st]), len(mature)


def live_frames(holdout: pd.DataFrame, online: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model inputs for the holdout period built from stream features: (honest, naive)."""
    merged = holdout.merge(online, on="TransactionID", how="left", validate="one_to_one")
    if merged["card_cnt_all"].isna().any():
        raise ValueError("online features are missing for some holdout rows; run `leakproof stream parity`")
    # The honest and naive models both have a column called amt_vs_card_mean with different meanings.
    naive = naive_serving_features(merged.drop(columns=["amt_vs_card_mean", "card_fraud_rate"]))
    return to_matrix(merged, HONEST_EXTRA), to_matrix(naive, NAIVE_EXTRA)


def run_headline(
    dev: pd.DataFrame | None = None,
    holdout: pd.DataFrame | None = None,
    online: pd.DataFrame | None = None,
    honest_offline: dict[str, float] | None = None,
    save_models: bool = True,
) -> list[dict[str, Any]]:
    dev = load_frame(("train", "valid")) if dev is None else dev
    dev = dev.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    holdout = load_frame(("holdout",)) if holdout is None else holdout
    online = pd.read_parquet(ONLINE_FEATURES_PATH) if online is None else online
    cutoff = float(holdout["TransactionDT"].min())

    naive_model, naive_offline = train_naive(dev)
    honest_model, mature_rows = train_honest(dev, cutoff)
    X_honest, X_naive = live_frames(holdout, online)
    y = holdout["isFraud"]
    naive_live, honest_live = score(naive_model, X_naive, y), score(honest_model, X_honest, y)

    if save_models:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": honest_model, "extra": HONEST_EXTRA}, MODEL_DIR / "honest.joblib")
        joblib.dump({"model": naive_model, "extra": NAIVE_EXTRA}, MODEL_DIR / "naive.joblib")

    def row(
        name: str, offline: dict[str, float] | None, live: dict[str, float], **more: Any
    ) -> dict[str, Any]:
        out: dict[str, Any] = {"model": name, "live_roc_auc": live["roc_auc"], "live_pr_auc": live["pr_auc"]}
        if offline:
            out |= {
                "offline_roc_auc": offline["roc_auc"],
                "offline_pr_auc": offline["pr_auc"],
                "roc_auc_gap": round(offline["roc_auc"] - live["roc_auc"], 4),
                "pr_auc_gap": round(offline["pr_auc"] - live["pr_auc"], 4),
            }
        return out | more

    return [
        row("naive", naive_offline, naive_live, train_rows=int(len(dev) * 0.70)),
        row("honest", honest_offline, honest_live, train_rows=mature_rows),
    ]


def honest_offline_estimate(path: Path = PROJECT_ROOT / "results" / "honest.json") -> dict[str, float] | None:
    """The validation-period score recorded before deployment (`leakproof run honest`)."""
    import json

    if not path.exists():
        return None
    for r in json.loads(path.read_text(encoding="utf-8")):
        if r["variant"] == "pit_honest":
            return {"roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"]}
    return None
