"""Choosing the alert threshold by money instead of by accuracy.

A transaction is sent to review when its score is at or above the threshold. The cost model:

    missed fraud   the transaction amount is lost
    any alert      a fixed review cost; a reviewed fraud is assumed to be stopped

    cost(threshold) = sum(amount of fraud below it) + review_cost * alerts

Thresholds are chosen on data available before deployment and then judged on the holdout
period's live scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from leakproof.config import PROJECT_ROOT

RESULTS_PATH = PROJECT_ROOT / "results" / "decisions.json"


def cost_curve(scores: Any, y: Any, amt: Any, review_cost: float) -> pd.DataFrame:
    """Cost and accuracy at every distinct score, plus +inf (alert on nothing)."""
    scores, y, amt = np.asarray(scores, float), np.asarray(y, int), np.asarray(amt, float)
    order = np.argsort(scores, kind="stable")
    s, fraud, a = scores[order], y[order], amt[order]
    thresholds = np.append(np.unique(s), np.inf)
    below = np.searchsorted(s, thresholds, side="left")  # rows that are not alerted
    missed_amt = np.concatenate([[0.0], np.cumsum(a * fraud)])[below]
    missed_n = np.concatenate([[0], np.cumsum(fraud)])[below]
    alerts = len(s) - below
    false_alerts = alerts - (fraud.sum() - missed_n)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "alerts": alerts,
            "cost": missed_amt + review_cost * alerts,
            "accuracy": 1 - (missed_n + false_alerts) / len(s),
        }
    )


def choose(curve: pd.DataFrame, by: str) -> float:
    """The threshold with the lowest cost (`by="cost"`) or the highest accuracy (`by="accuracy"`)."""
    row = curve["cost"].idxmin() if by == "cost" else curve["accuracy"].idxmax()
    return float(curve.loc[row, "threshold"])


def evaluate(scores: Any, y: Any, amt: Any, threshold: float, review_cost: float) -> dict[str, Any]:
    scores, y, amt = np.asarray(scores, float), np.asarray(y, int), np.asarray(amt, float)
    alert, fraud = scores >= threshold, y == 1
    alerts, caught = int(alert.sum()), int((alert & fraud).sum())
    missed_amt = float(amt[fraud & ~alert].sum())
    cost = missed_amt + review_cost * alerts
    return {
        "threshold": None if np.isinf(threshold) else round(threshold, 4),
        "alerts": alerts,
        "alert_rate": round(alerts / len(y), 4),
        "precision": round(caught / alerts, 4) if alerts else None,
        "recall": round(caught / int(fraud.sum()), 4),
        "fraud_caught_amt": round(float(amt[fraud & alert].sum())),
        "fraud_missed_amt": round(missed_amt),
        "review_cost": round(review_cost * alerts),
        "total_cost": round(cost),
        "cost_per_1k_tx": round(cost / len(y) * 1000, 1),
    }


def run_decisions(
    honest_select: pd.DataFrame,
    naive_select: pd.DataFrame,
    holdout: pd.DataFrame,
    review_cost: float,
) -> list[dict[str, Any]]:
    """One row per policy, judged on the holdout period. The frames need `isFraud` and
    `TransactionAmt`; the selection frames need `score`, the holdout `score_honest`/`score_naive`."""
    y, amt = holdout["isFraud"], holdout["TransactionAmt"]

    def promised(sel: pd.DataFrame, thr: float) -> float:
        return evaluate(sel["score"], sel["isFraud"], sel["TransactionAmt"], thr, review_cost)[
            "cost_per_1k_tx"
        ]

    rows = [{"policy": "no_model", **evaluate(holdout["score_honest"], y, amt, np.inf, review_cost)}]
    for name, sel in (("honest", honest_select), ("naive", naive_select)):
        live = holdout[f"score_{name}"]
        curve = cost_curve(sel["score"], sel["isFraud"], sel["TransactionAmt"], review_cost)
        picks = {"cost": choose(curve, "cost")}
        if name == "honest":
            picks = {"accuracy": choose(curve, "accuracy"), **picks}
        for by, thr in picks.items():
            rows.append(
                {
                    "policy": f"{name}_{by}_threshold",
                    **evaluate(live, y, amt, thr, review_cost),
                    "promised_cost_per_1k_tx": promised(sel, thr),
                }
            )
        hindsight = choose(cost_curve(live, y, amt, review_cost), "cost")
        rows.append({"policy": f"{name}_hindsight", **evaluate(live, y, amt, hindsight, review_cost)})
    return rows
