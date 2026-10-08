"""Leakproof dashboard: the leakage gap, the money, drift and delayed labels.

    uv run streamlit run ui/app.py

Reads the JSON files in results/, so it works without Docker or the dataset. The last section
calls the scoring API if one is running (`uv run leakproof serve run`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parents[1] / "results"
API_URL = os.environ.get("LEAKPROOF_API", "http://127.0.0.1:8000")
BLUE, ORANGE = "#2a78d6", "#eb6834"  # first two slots of a colourblind-checked palette

LADDER_LABELS = {
    "time_split_base": "1. time split, base features",
    "random_base": "2. random split",
    "random_global_aggs": "3. + aggregates over all rows",
    "random_target_enc": "4. + fraud rate over all rows",
}
POLICY_LABELS = {
    "no_model": "No model",
    "honest_accuracy_threshold": "Honest, accuracy-chosen threshold",
    "honest_cost_threshold": "Honest, cost-chosen threshold",
    "honest_hindsight": "Honest, best threshold in hindsight",
    "naive_cost_threshold": "Careless, cost-chosen threshold",
    "naive_hindsight": "Careless, best threshold in hindsight",
}


def load(name: str) -> Any:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def hbar(df: pd.DataFrame, value: str, label: str, title: str, fmt: str) -> alt.LayerChart:
    """Horizontal bars in the order given, with the value written at the end of each bar."""
    base = alt.Chart(df).encode(
        y=alt.Y(f"{label}:N", sort=None, title=None, axis=alt.Axis(labelLimit=320)),
        x=alt.X(f"{value}:Q", title=title, axis=alt.Axis(tickCount=6)),
        tooltip=[label, alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    bars = base.mark_bar(color=BLUE, cornerRadiusEnd=4, height={"band": 0.6})
    text = base.mark_text(align="left", dx=4).encode(text=alt.Text(f"{value}:Q", format=fmt))
    return bars + text


st.set_page_config(page_title="Leakproof", layout="wide")
st.title("Leakproof")
st.caption(
    "Fraud detection on IEEE-CIS that measures its own data leakage: the same model, scored offline "
    "and then on features computed by a live stream."
)

headline, decisions = load("headline"), load("decisions")
if headline is None:
    st.error("No results found. Run the pipeline first (see the README).")
    st.stop()

# -- 1. the gap ------------------------------------------------------------------------------
st.header("Offline score against live score")
by_model = {r["model"]: r for r in headline}
cols = st.columns(4)
for col, (model, name) in zip(cols[::2], (("naive", "Careless"), ("honest", "Honest")), strict=True):
    r = by_model[model]
    col.metric(f"{name} model, live PR-AUC", r["live_pr_auc"], f"{-r['pr_auc_gap']:+.3f} vs offline")
for col, (model, name) in zip(cols[1::2], (("naive", "Careless"), ("honest", "Honest")), strict=True):
    r = by_model[model]
    col.metric(f"{name} model, live ROC-AUC", r["live_roc_auc"], f"{-r['roc_auc_gap']:+.3f} vs offline")

long = pd.DataFrame(
    [
        {"model": name, "where": where.capitalize(), "metric": label, "score": by_model[m][f"{where}_{key}"]}
        for m, name in (("naive", "Careless"), ("honest", "Honest"))
        for where in ("offline", "live")
        for key, label in (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"))
    ]
)
for col, metric in zip(st.columns(2), ("PR-AUC", "ROC-AUC"), strict=True):
    chart = (
        alt.Chart(long[long["metric"] == metric], title=metric)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("model:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("where:N", sort=["Offline", "Live"]),
            y=alt.Y("score:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "where:N",
                title=None,
                scale=alt.Scale(domain=["Offline", "Live"], range=[BLUE, ORANGE]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["model", "where", alt.Tooltip("score:Q", format=".4f")],
        )
        .properties(height=280)
    )
    col.altair_chart(chart, width="stretch")
st.caption(
    "Offline: the score each pipeline reported for itself before deployment. "
    "Live: the holdout period, scored on stream features."
)

ladder = load("naive")
if ladder:
    st.subheader("Where the careless score comes from")
    df = pd.DataFrame(ladder).assign(step=lambda d: d["variant"].map(LADDER_LABELS))
    st.altair_chart(
        hbar(df, "roc_auc", "step", "Offline ROC-AUC", ".4f").properties(height=200), width="stretch"
    )
    st.caption("Each step adds one careless habit. The model and its settings never change.")

# -- 2. the money ----------------------------------------------------------------------------
if decisions:
    st.header("What a threshold costs")
    rows = {r["policy"]: r for r in decisions}
    chosen, by_accuracy = rows["honest_cost_threshold"], rows["honest_accuracy_threshold"]
    careless = rows["naive_cost_threshold"]
    a, b, c = st.columns(3)
    a.metric("Cost with the cost-chosen threshold", f"${chosen['total_cost']:,}")
    b.metric(
        "Saved against the accuracy-chosen threshold",
        f"${by_accuracy['total_cost'] - chosen['total_cost']:,}",
    )
    c.metric(
        "Careless model, cost per 1,000 transactions",
        f"${careless['cost_per_1k_tx']:,.0f}",
        f"promised ${careless['promised_cost_per_1k_tx']:,.0f}",
        delta_color="off",
    )
    df = pd.DataFrame(decisions).assign(policy=lambda d: d["policy"].map(POLICY_LABELS))
    st.altair_chart(
        hbar(df, "total_cost", "policy", "Total cost on the holdout period (USD)", "$,.0f").properties(
            height=260
        ),
        width="stretch",
    )
    st.caption(
        "Cost = fraud that was not alerted (the amount is lost) + a fixed cost per alert reviewed. "
        "Thresholds were chosen before the holdout period, except the two hindsight rows."
    )
    with st.expander("All numbers"):
        st.dataframe(df, hide_index=True)

# -- 3. monitoring ---------------------------------------------------------------------------
drift, daily = load("drift"), load("daily")
if drift:
    st.header("Feature drift")
    features = pd.DataFrame([r for r in drift if r["feature"] != "model_score"])
    score_psi = next(r["psi"] for r in drift if r["feature"] == "model_score")
    counts = features["level"].value_counts()
    a, b, c = st.columns(3)
    a.metric("Model score PSI", score_psi)
    b.metric("Features with PSI of 0.25 or more", int(counts.get("major", 0)))
    c.metric("Features with PSI from 0.1 to 0.25", int(counts.get("moderate", 0)))
    st.altair_chart(
        hbar(
            features.head(12), "psi", "feature", "PSI, development period against holdout period", ".3f"
        ).properties(height=320),
        width="stretch",
    )
    st.caption("Rule of thumb: below 0.1 is stable, 0.1 to 0.25 is worth a look, above 0.25 is a real shift.")

if daily:
    st.header("Daily recall, and what delayed labels let you see")
    df = pd.DataFrame(daily)
    lines = df.melt("day", ["recall", "recall_known"], "view", "value").replace(
        {"recall": "Final truth", "recall_known": "Known on the last day"}
    )
    chart = (
        alt.Chart(lines.dropna())
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60))
        .encode(
            x=alt.X("day:Q", title="Day of the holdout period"),
            y=alt.Y("value:Q", title="Recall of fraud", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "view:N",
                title=None,
                scale=alt.Scale(domain=["Final truth", "Known on the last day"], range=[BLUE, ORANGE]),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=["day", "view", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "A fraud label takes about two weeks to arrive and a legitimate one 30 days. For recent days "
        "only part of the fraud is confirmed, so the orange line is noisier and then stops."
    )
    with st.expander("Daily table"):
        st.dataframe(df, hide_index=True)

# -- 4. live ---------------------------------------------------------------------------------
st.header("Score a transaction")
st.caption(f"Calls the scoring API at {API_URL}. Start it with `uv run leakproof serve run`.")
example = {
    "TransactionID": 1,
    "TransactionDT": 15_000_000,
    "TransactionAmt": 250.0,
    "card1": 9500,
    "ProductCD": "W",
}
body = st.text_area("Transaction (JSON)", json.dumps(example, indent=2), height=180)
if st.button("Score"):
    import httpx

    try:
        response = httpx.post(f"{API_URL}/score", json=json.loads(body), timeout=5)
        st.json(response.json())
    except (httpx.HTTPError, ValueError) as e:
        st.warning(f"Could not score: {e}")
