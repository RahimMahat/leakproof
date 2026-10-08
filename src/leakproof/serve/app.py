"""Online scoring service.

    POST /score   score one transaction on features read from Redis, then record it
    GET  /health

A request is handled in the same order as the stream consumer: read the card's features as of
the transaction's time, score, and only then add the transaction to the card's history.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from leakproof.features.base import card_id_from_row
from leakproof.features.online import OnlineFeatureStore
from leakproof.serve.scorer import FastScorer


class Transaction(BaseModel):
    """The three required fields; every other dataset column is accepted as an extra field."""

    model_config = ConfigDict(extra="allow")

    TransactionID: int
    TransactionDT: int = Field(ge=0)
    TransactionAmt: float = Field(gt=0)


class Score(BaseModel):
    transaction_id: int
    card_id: str
    score: float
    card_cnt_24h: float
    card_fraud_cnt: float
    latency_ms: float


def create_app(store: OnlineFeatureStore, model: Any, extra: list[str]) -> FastAPI:
    app = FastAPI(title="Leakproof scoring", version="0.1.0")
    scorer = FastScorer(model, extra)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/score", response_model=Score)
    def score(tx: Transaction) -> Score:
        t0 = time.perf_counter()
        row = tx.model_dump()
        card_id = card_id_from_row(row)
        raw = store.raw_features(card_id, tx.TransactionDT)
        p = scorer.score(scorer.features(row, raw))
        store.apply_transaction(card_id, tx.TransactionID, tx.TransactionDT, tx.TransactionAmt)
        return Score(
            transaction_id=tx.TransactionID,
            card_id=card_id,
            score=p,
            card_cnt_24h=raw["card_cnt_24h"],
            card_fraud_cnt=raw["card_fraud_cnt"],
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    return app


def app_from_env() -> FastAPI:
    """Factory for `uvicorn --factory`: the honest model and the Redis from config."""
    import joblib
    import redis

    from leakproof.config import REDIS_URL
    from leakproof.model.final import MODEL_DIR

    bundle = joblib.load(MODEL_DIR / "honest.joblib")
    return create_app(OnlineFeatureStore(redis.Redis.from_url(REDIS_URL)), bundle["model"], bundle["extra"])
