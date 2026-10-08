"""One model configuration for every experiment, so only the data handling differs."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from leakproof.config import SEED

PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 2000,  # upper bound; early stopping picks the real number
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_lambda": 1.0,
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 50


def fit(
    X_train: pd.DataFrame, y_train: pd.Series, X_stop: pd.DataFrame, y_stop: pd.Series
) -> lgb.LGBMClassifier:
    """Train with early stopping on (X_stop, y_stop). That set is never the reported test set."""
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_X=X_stop,
        eval_y=y_stop,
        eval_metric="auc",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def score(model: lgb.LGBMClassifier, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    p = np.asarray(model.predict_proba(X))[:, 1]
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
    }
