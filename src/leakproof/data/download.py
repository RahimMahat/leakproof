"""Download the IEEE-CIS Fraud Detection files from Kaggle into data/raw."""

from __future__ import annotations

import shutil
from pathlib import Path

from leakproof.config import KAGGLE_DATASET, RAW_DIR, RAW_FILES


def download_raw(force: bool = False) -> Path:
    if not force and all((RAW_DIR / f).exists() for f in RAW_FILES):
        return RAW_DIR
    import kagglehub

    cache = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        shutil.copy2(cache / name, RAW_DIR / name)
    return RAW_DIR
