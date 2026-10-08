"""The leakage ladder: the same model, with one more careless step at each rung.

Everything here uses only the development data (the train and valid periods). The holdout period
is the "future" and stays untouched until the live replay.

    time_split_base      train on the train period, test on the valid period, base features
    random_base          random 70/15/15 split of the development data, base features
    random_global_aggs   + card aggregates computed over all development rows
    random_target_enc    + card fraud rate computed over all development rows
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from leakproof.config import PROJECT_ROOT, SEED
from leakproof.features.base import load_frame, to_matrix
from leakproof.features.naive import (
    GLOBAL_AGG_FEATURES,
    TARGET_ENC_FEATURES,
    add_global_aggregates,
    add_target_encoding,
)
from leakproof.model.train import fit, score

RESULTS_PATH = PROJECT_ROOT / "results" / "naive.json"
STOP_FRAC = 0.15  # share of the training rows held back for early stopping


def random_split(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row positions for a shuffled 70/15/15 train / early-stop / test split."""
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(n * 0.70), int(n * 0.85)
    return idx[:a], idx[a:b], idx[b:]


def run_variant(
    df: pd.DataFrame, extra: list[str], tr: np.ndarray, st: np.ndarray, te: np.ndarray
) -> dict[str, Any]:
    X, y = to_matrix(df, extra), df["isFraud"]
    model = fit(X.iloc[tr], y.iloc[tr], X.iloc[st], y.iloc[st])
    return {**score(model, X.iloc[te], y.iloc[te]), "trees": int(model.best_iteration_), "test_rows": len(te)}


def run_ladder(df: pd.DataFrame | None = None) -> list[dict[str, Any]]:
    df = load_frame(("train", "valid")) if df is None else df
    df = df.reset_index(drop=True)
    results: list[dict[str, Any]] = []

    # Reference: split by time. The last part of the train period is used for early stopping.
    train_pos = np.flatnonzero(df["period"] == "train")
    cut = int(len(train_pos) * (1 - STOP_FRAC))
    results.append(
        {
            "variant": "time_split_base",
            "split": "time",
            **run_variant(df, [], train_pos[:cut], train_pos[cut:], np.flatnonzero(df["period"] == "valid")),
        }
    )

    tr, st, te = random_split(len(df))
    results.append({"variant": "random_base", "split": "random", **run_variant(df, [], tr, st, te)})

    aggs = add_global_aggregates(df)
    results.append(
        {
            "variant": "random_global_aggs",
            "split": "random",
            **run_variant(aggs, GLOBAL_AGG_FEATURES, tr, st, te),
        }
    )

    enc = add_target_encoding(aggs)
    results.append(
        {
            "variant": "random_target_enc",
            "split": "random",
            **run_variant(enc, [*GLOBAL_AGG_FEATURES, *TARGET_ENC_FEATURES], tr, st, te),
        }
    )
    return results


def save(results: list[dict[str, Any]], path: Path = RESULTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return path


def log_to_mlflow(results: list[dict[str, Any]], experiment: str = "leakage-ladder") -> None:
    import mlflow

    from leakproof.config import DATA_DIR
    from leakproof.model.train import PARAMS

    mlflow.set_tracking_uri(f"sqlite:///{(DATA_DIR / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(experiment)
    for r in results:
        with mlflow.start_run(run_name=r["variant"]):
            mlflow.log_params({**PARAMS, "split": r["split"]})
            mlflow.log_metrics({k: r[k] for k in ("roc_auc", "pr_auc", "trees")})
