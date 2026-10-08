"""Fast single-transaction scoring.

Scoring one row through pandas costs ~100 ms (building a typed one-row DataFrame, then LightGBM
re-aligning its categorical columns). This path builds a plain float array instead, encodes each
categorical with the category list the model stored at training time, and calls the booster
directly. The replay check compares these scores with batch scores for the same transactions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from leakproof.features.base import CATEGORICAL, NUMERIC
from leakproof.features.spec import derive

NAN = float("nan")


class FastScorer:
    def __init__(self, model: Any, extra: list[str]) -> None:
        self.booster = model.booster_
        self.numeric = [*NUMERIC]
        self.extra = list(extra)
        stored = self.booster.pandas_categorical
        if stored is None or len(stored) != len(CATEGORICAL):
            raise ValueError("model was not trained on the expected categorical columns")
        # LightGBM encodes a category as its position in the training-time category list.
        self.codes = [{value: float(i) for i, value in enumerate(cats)} for cats in stored]
        if self.booster.feature_name() != [*NUMERIC, *CATEGORICAL, *self.extra]:
            raise ValueError("model feature order differs from NUMERIC + CATEGORICAL + extra")

    def features(self, row: dict[str, Any], raw: dict[str, float]) -> dict[str, Any]:
        """The transaction's fields plus raw and derived card features."""
        f = np.float64
        derived = derive(
            f(row["TransactionDT"]),
            f(row["TransactionAmt"]),
            f(raw["card_last_t"]),
            f(raw["card_first_t"]),
            f(raw["card_amt_mean_all"]),
            f(raw["card_fraud_cnt"]),
            f(raw["card_labeled_cnt"]),
        )
        return {**row, **raw, **{k: float(v) for k, v in derived.items()}}

    def score(self, features: dict[str, Any]) -> float:
        x = np.empty(len(self.numeric) + len(CATEGORICAL) + len(self.extra))
        i = 0
        for c in self.numeric:
            x[i] = _num(features.get(c))
            i += 1
        for c, codes in zip(CATEGORICAL, self.codes, strict=True):
            x[i] = codes.get(features.get(c), NAN)  # unseen or missing category -> missing
            i += 1
        for c in self.extra:
            x[i] = _num(features.get(c))
            i += 1
        return float(self.booster.predict(x.reshape(1, -1), num_threads=1)[0])


def _num(v: Any) -> float:
    return NAN if v is None else float(v)
