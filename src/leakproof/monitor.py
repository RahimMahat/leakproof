"""Monitoring: feature drift (PSI) and daily performance under delayed labels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leakproof.config import DAY, PROJECT_ROOT

DRIFT_PATH = PROJECT_ROOT / "results" / "drift.json"
DAILY_PATH = PROJECT_ROOT / "results" / "daily.json"
EPS = 1e-4  # floor for an empty bin, so the log is defined
MODERATE, MAJOR = 0.1, 0.25  # the usual rule-of-thumb PSI levels


def _shares(ref: pd.Series, cur: pd.Series, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Share of rows per bin for both samples. Missing values get their own bin."""
    if pd.api.types.is_numeric_dtype(ref):
        r, c = ref.to_numpy(float), cur.to_numpy(float)
        known = r[~np.isnan(r)]
        qs = np.quantile(known, np.linspace(0, 1, bins + 1)[1:-1]) if len(known) else []
        edges = np.unique(qs)

        def share(x: np.ndarray) -> np.ndarray:
            codes = np.where(np.isnan(x), 0, np.searchsorted(edges, x, side="right") + 1)
            return np.bincount(codes, minlength=len(edges) + 2) / len(x)

        return share(r), share(c)
    a = ref.astype(object).where(ref.notna(), "<missing>").value_counts(normalize=True)
    b = cur.astype(object).where(cur.notna(), "<missing>").value_counts(normalize=True)
    keys = a.index.union(b.index)
    return a.reindex(keys, fill_value=0).to_numpy(), b.reindex(keys, fill_value=0).to_numpy()


def psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """Population stability index of `cur` against `ref`. Numeric columns are binned on the
    reference deciles; other columns are compared category by category."""
    p, q = (np.clip(s, EPS, None) for s in _shares(ref, cur, bins))
    return float(np.sum((q - p) * np.log(q / p)))


def drift_report(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """PSI for every column the two frames share, largest first."""
    rows = [{"feature": c, "psi": round(psi(ref[c], cur[c]), 4)} for c in ref.columns if c in cur.columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    out["level"] = np.select([out["psi"] >= MAJOR, out["psi"] >= MODERATE], ["major", "moderate"], "stable")
    return out


def daily_report(scored: pd.DataFrame, threshold: float, as_of: float) -> pd.DataFrame:
    """One row per scoring day: what was true in the end, and what could be seen at `as_of`.

    `scored` needs TransactionDT, label_available_at, isFraud and score. Columns ending in
    `_known` use only labels that had arrived by `as_of`. A legitimate label takes 30 days, so
    for recent days precision can't be computed yet; recall among confirmed fraud can.
    """
    d = scored.assign(
        day=((scored["TransactionDT"] - scored["TransactionDT"].min()) // DAY).astype(int),
        alert=scored["score"] >= threshold,
        fraud=scored["isFraud"] == 1,
        known=scored["label_available_at"] <= as_of,
    )
    d = d.assign(
        caught=d["alert"] & d["fraud"],
        fraud_known=d["fraud"] & d["known"],
        caught_known=d["alert"] & d["fraud"] & d["known"],
    )
    g = d.groupby("day")[["alert", "fraud", "caught", "known", "fraud_known", "caught_known"]].sum()
    n = d.groupby("day").size()
    return pd.DataFrame(
        {
            "day": g.index,
            "transactions": n.to_numpy(),
            "alerts": g["alert"].to_numpy(),
            "fraud": g["fraud"].to_numpy(),
            "precision": (g["caught"] / g["alert"]).round(3).to_numpy(),
            "recall": (g["caught"] / g["fraud"]).round(3).to_numpy(),
            "labels_known": (g["known"] / n).round(3).to_numpy(),
            "fraud_known": g["fraud_known"].to_numpy(),
            "recall_known": (g["caught_known"] / g["fraud_known"]).round(3).to_numpy(),
        }
    )


def to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-ready rows (NaN becomes null)."""
    return df.astype(object).where(df.notna(), None).to_dict("records")
