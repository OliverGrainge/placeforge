from __future__ import annotations

import os
from pathlib import Path


def _require_env(var: str) -> Path:
    value = os.environ.get(var)
    if not value:
        raise EnvironmentError(
            f"Required environment variable {var!r} is not set. "
            f"Add it to your .env file or shell environment."
        )
    return Path(value)


def raw_dir() -> Path:
    """Root directory for raw downloaded datasets (PLACEFORGE_RAW_DIR)."""
    return _require_env("PLACEFORGE_RAW_DIR")


def feature_store_dir() -> Path:
    """Root directory for computed feature artifacts such as embeddings (PLACEFORGE_FEATURE_STORE_DIR)."""
    return _require_env("PLACEFORGE_FEATURE_STORE_DIR")


def processed_dir() -> Path:
    """Root directory for processed train/val/test datasets (PLACEFORGE_PROCESSED_DIR)."""
    return _require_env("PLACEFORGE_PROCESSED_DIR")
