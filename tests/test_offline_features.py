"""Point-in-time features: correct values, and provably blind to the future."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from leakproof.features.offline import build_offline_features
from leakproof.features.spec import DAY, HOUR, LABEL_FEATURES, PIT_FEATURES, WEEK

FEATURES = [*PIT_FEATURES, *LABEL_FEATURES]


def synthetic(n: int = 1500, cards: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dt = np.sort(rng.integers(0, 40 * DAY, n))
    dt[100:110] = dt[100]  # force same-second transactions
    fraud = (rng.random(n) < 0.15).astype(int)
    delay = np.where(fraud == 1, rng.integers(1, 20, n), 30) * DAY
    return pd.DataFrame(
        {
            "TransactionID": rng.permutation(n) + 10_000,
            "TransactionDT": dt,
            "TransactionAmt": rng.gamma(2.0, 40.0, n).round(2),
            "card_id": rng.integers(0, cards, n).astype(str),
            "isFraud": fraud,
            "label_available_at": (dt + delay).astype(float),
        }
    )


def reference(df: pd.DataFrame) -> pd.DataFrame:
    """Slow, obviously-correct implementation of the time rule in plain Python."""
    rows = []
    for r in df.itertuples():
        card = df[df["card_id"] == r.card_id]
        prior = card[card["TransactionDT"] <= r.TransactionDT - 1]
        amt = prior["TransactionAmt"]

        def within(seconds: int, prior: pd.DataFrame = prior, r=r) -> pd.DataFrame:
            return prior[prior["TransactionDT"] >= r.TransactionDT - seconds]

        known = card[card["label_available_at"] <= r.TransactionDT]
        labeled, frauds = len(known), int(known["isFraud"].sum())
        rows.append(
            {
                "TransactionID": r.TransactionID,
                "card_cnt_1h": len(within(HOUR)),
                "card_cnt_24h": len(within(DAY)),
                "card_cnt_7d": len(within(WEEK)),
                "card_amt_sum_24h": within(DAY)["TransactionAmt"].sum(),
                "card_amt_sum_7d": within(WEEK)["TransactionAmt"].sum(),
                "card_cnt_all": len(prior),
                "card_amt_mean_all": amt.mean() if len(prior) else np.nan,
                "card_amt_std_all": amt.std() if len(prior) > 1 else np.nan,
                "secs_since_last_tx": r.TransactionDT - prior["TransactionDT"].max()
                if len(prior)
                else np.nan,
                "card_age_s": r.TransactionDT - prior["TransactionDT"].min() if len(prior) else np.nan,
                "amt_vs_card_mean": r.TransactionAmt / amt.mean() if len(prior) else np.nan,
                "card_labeled_cnt": labeled,
                "card_fraud_cnt": frauds,
                "card_fraud_rate": frauds / labeled if labeled else np.nan,
            }
        )
    return pd.DataFrame(rows)


def aligned(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = a.set_index("TransactionID").sort_index()[FEATURES].astype(float)
    b = b.set_index("TransactionID").sort_index()[FEATURES].astype(float)
    return a, b


def test_matches_the_reference_implementation():
    df = synthetic(600, cards=25)
    got, want = aligned(build_offline_features(df), reference(df))
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-9, atol=1e-9)


def test_first_transaction_of_a_card_knows_nothing():
    df = synthetic()
    feats = build_offline_features(df).merge(df[["TransactionID", "card_id", "TransactionDT"]])
    first = feats.sort_values("TransactionDT").groupby("card_id").head(1)
    assert (first["card_cnt_all"] == 0).all() and (first["card_labeled_cnt"] == 0).all()
    assert first[["card_amt_mean_all", "secs_since_last_tx", "card_fraud_rate"]].isna().all().all()


@pytest.mark.parametrize("cut_day", [5, 17, 31])
def test_the_future_cannot_change_the_past(cut_day):
    """Features for transactions up to time T are identical whether or not anything after T exists."""
    df = synthetic()
    cut = cut_day * DAY
    past_ids = df.loc[df["TransactionDT"] <= cut, "TransactionID"]
    full = build_offline_features(df)
    truncated = build_offline_features(df[df["TransactionDT"] <= cut])
    got, want = aligned(full[full["TransactionID"].isin(past_ids)], truncated)
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-12, atol=1e-12)


def test_changing_future_rows_and_unarrived_labels_changes_nothing():
    df = synthetic()
    cut = 20 * DAY
    tampered = df.copy()
    future = tampered["TransactionDT"] > cut
    tampered.loc[future, "TransactionAmt"] *= 100
    tampered.loc[future, "isFraud"] = 1
    # flip every label that has not arrived by the cut, including labels of past transactions
    unarrived = tampered["label_available_at"] > cut
    tampered.loc[unarrived, "isFraud"] = 1 - tampered.loc[unarrived, "isFraud"]
    past_ids = df.loc[~future, "TransactionID"]
    a = build_offline_features(df)
    b = build_offline_features(tampered)
    got, want = aligned(a[a["TransactionID"].isin(past_ids)], b[b["TransactionID"].isin(past_ids)])
    pd.testing.assert_frame_equal(got, want)


def test_same_second_transactions_do_not_see_each_other():
    df = synthetic()
    feats = build_offline_features(df).merge(df[["TransactionID", "card_id", "TransactionDT"]])
    dup = feats[feats.duplicated(["card_id", "TransactionDT"], keep=False)]
    if len(dup):  # every transaction in a same-second group has the same history
        assert dup.groupby(["card_id", "TransactionDT"])["card_cnt_all"].nunique().max() == 1


def test_instant_labels_see_more_than_delayed_labels():
    df = synthetic()
    delayed = build_offline_features(df)["card_labeled_cnt"].sum()
    instant = build_offline_features(df, instant_labels=True)["card_labeled_cnt"].sum()
    assert instant > delayed
