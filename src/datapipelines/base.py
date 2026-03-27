from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

from .steps.base import BaseStep


def _hash_context(context: dict[str, Any]) -> str:
    """Compute a deterministic hash of the pipeline context."""
    h = hashlib.sha256()
    for key in sorted(context.keys()):
        h.update(key.encode())
        value = context[key]
        if isinstance(value, pd.DataFrame):
            row_hashes = pd.util.hash_pandas_object(value)
            h.update(row_hashes.values.tobytes())
        else:
            h.update(pickle.dumps(value))
    return h.hexdigest()[:16]


def _cache_key(step_name: str, params: dict[str, Any], context: dict[str, Any]) -> str:
    """Compute a cache key from step identity, its params, and the input context."""
    h = hashlib.sha256()
    h.update(step_name.encode())
    h.update(json.dumps(params, sort_keys=True).encode())
    h.update(_hash_context(context).encode())
    return h.hexdigest()[:16]


def _step_cache_dir() -> Path:
    return Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"]) / "cache" / "steps"


class Pipeline:
    def __init__(self, name: str, steps: Iterable[BaseStep]) -> None:
        self.name = name
        self.steps = list(steps)

    def run(self, context: dict[str, Any] | None = None, use_cache: bool = False) -> dict[str, Any]:
        context = context or {}

        for step in self.steps:
            name = type(step).__name__
            params = step.cache_params()

            # Try loading from cache
            if use_cache and params is not None:
                key = _cache_key(name, params, context)
                cache_path = _step_cache_dir() / f"{key}.pkl"
                if cache_path.exists():
                    with tqdm(total=1, desc=f"{name} (cached)", leave=True) as pbar:
                        context = pickle.loads(cache_path.read_bytes())
                        pbar.update(1)
                    continue

            # Run step
            with tqdm(total=1, desc=name, leave=True, disable=not step.show_pbar) as pbar:
                step.pbar = pbar
                result = step.run(context)
                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
            step.pbar = None
            if result is not None:
                context = result

            # Save to cache
            if use_cache and params is not None:
                cache_dir = _step_cache_dir()
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(pickle.dumps(context))

        return context
