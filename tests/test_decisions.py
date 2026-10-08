"""Cost-based threshold, PSI and the delayed-label daily report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakproof.config import DAY
from leakproof.model.threshold import choose, cost_curve, evaluate, run_decisions
from leakproof.monitor import daily_report, drift_report, psi

SCORES = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
FRAUD = np.array([0, 0, 1, 0, 1, 1])
AMT = np.array([50.0, 50.0, 400.0, 50.0, 20.0, 300.0])


def test_cost_threshold_follows_the_money_and_accuracy_does_not():
    curve = cost_curve(SCORES, FRAUD, AMT, review_cost=10)
    # Catching the 400 fraud at 0.3 costs one extra false alert (10); missing it costs 400.
    assert choose(curve, "cost") == 0.3
    assert curve["cost"].min() == 40  # 4 alerts, nothing missed
    assert choose(curve, "accuracy") in (0.3, 0.7)  # both get 5 of 6 right
    assert np.isinf(curve["threshold"].iloc[-1]) and curve["cost"].iloc[-1] == 720  # alert on nothing


def test_expensive_reviews_raise_the_threshold():
    assert choose(cost_curve(SCORES, FRAUD, AMT, review_cost=250), "cost") == 0.9


def test_evaluate_adds_up():
    r = evaluate(SCORES, FRAUD, AMT, 0.6, review_cost=10)
    assert (r["alerts"], r["precision"], r["recall"]) == (3, 0.6667, 0.6667)
    assert (r["fraud_caught_amt"], r["fraud_missed_amt"], r["review_cost"], r["total_cost"]) == (
        320,
        400,
        30,
        430,
    )


def test_run_decisions_rows_and_hindsight_is_a_lower_bound():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.05
    frame = pd.DataFrame(
        {
            "isFraud": y.astype(int),
            "TransactionAmt": rng.lognormal(4, 1, 4000),
            "score": rng.random(4000) * 0.5 + y * 0.4,
        }
    )
    sel, hold = frame.iloc[:2000], frame.iloc[2000:]
    hold = hold.assign(score_honest=hold["score"], score_naive=rng.random(2000))
    rows = {r["policy"]: r for r in run_decisions(sel, sel, hold, review_cost=5)}
    assert list(rows) == [
        "no_model", "honest_accuracy_threshold", "honest_cost_threshold", "honest_hindsight",
        "naive_cost_threshold", "naive_hindsight",
    ]  # fmt: skip
    assert rows["no_model"]["alerts"] == 0 and rows["no_model"]["recall"] == 0
    assert rows["honest_hindsight"]["total_cost"] <= rows["honest_cost_threshold"]["total_cost"]
    assert rows["honest_cost_threshold"]["total_cost"] < rows["no_model"]["total_cost"]


def test_psi():
    rng = np.random.default_rng(1)
    a, b = pd.Series(rng.normal(size=20_000)), pd.Series(rng.normal(size=20_000))
    assert psi(a, b) < 0.01
    assert psi(a, b + 1) > 0.25
    with_gaps = b.mask(rng.random(len(b)) < 0.3)
    assert psi(a, with_gaps) > 0.25  # a jump in the missing rate is drift
    cats = pd.Series(["x"] * 900 + ["y"] * 100, dtype="category")
    assert psi(cats, cats) == 0
    assert psi(cats, pd.Series(["x"] * 500 + ["z"] * 500)) > 0.25  # a category never seen before
    report = drift_report(pd.DataFrame({"same": a, "moved": a}), pd.DataFrame({"same": b, "moved": b + 1}))
    assert report["feature"].tolist() == ["moved", "same"] and report["level"].tolist() == ["major", "stable"]


def test_daily_report_separates_final_truth_from_what_is_known():
    scored = pd.DataFrame(
        {
            "TransactionDT": [0, 10, 20, DAY + 5, DAY + 6],
            "isFraud": [1, 1, 0, 1, 0],
            "score": [0.9, 0.1, 0.8, 0.9, 0.1],
            "label_available_at": [2 * DAY, 9 * DAY, 30 * DAY, 9 * DAY, 31 * DAY],
        }
    )
    day0, day1 = daily_report(scored, threshold=0.5, as_of=3 * DAY).to_dict("records")
    assert (day0["transactions"], day0["alerts"], day0["fraud"]) == (3, 2, 2)
    assert (day0["precision"], day0["recall"]) == (0.5, 0.5)
    # Only the caught fraud is confirmed so far, so the early view is too rosy.
    assert (day0["fraud_known"], day0["recall_known"], day0["labels_known"]) == (1, 1.0, 0.333)
    assert (
        day1["fraud_known"] == 0 and np.isnan(day1["recall_known"]) and day1["recall"] == pytest.approx(1.0)
    )
