"""Paths and constants. Override the data location with LEAKPROOF_DATA_DIR."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LEAKPROOF_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
DATASET_PATH = DATA_DIR / "transactions.parquet"

# A public re-upload of the IEEE-CIS Fraud Detection competition files. It downloads without a
# Kaggle login; the competition's own terms (non-commercial, research/education use) still apply.
KAGGLE_DATASET = "lnasiri007/ieeecis-fraud-detection"
EXPECTED_ROWS = 590_540  # train_transaction.csv, per the competition
RAW_FILES = ("train_transaction.csv", "train_identity.csv")  # the test_* files have no labels

SEED = 42
DAY = 86_400  # TransactionDT is in seconds from an undisclosed start

KAFKA_BOOTSTRAP = os.environ.get("LEAKPROOF_KAFKA", "localhost:9092")
REDIS_URL = os.environ.get("LEAKPROOF_REDIS", "redis://localhost:6379/0")

# Cost of sending one alert to a human reviewer, in the dataset's currency (USD).
REVIEW_COST = float(os.environ.get("LEAKPROOF_REVIEW_COST", "5"))
