from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseStep

# Maximum number of chunk files held in memory at once.  Each chunk is at most
# batch_size × embed_dim × 4 bytes ≈ 0.2 MB (batch=64, vitb14_reg=768-dim).
# 256 chunks caps the cache at roughly 50 MB in the typical case.
_MAX_CACHED_CHUNKS = 256


class FilterVisualOverlapStep(BaseStep):
    """Discard intra-place images that are visual outliers.

    For every ``place_id`` group the step:

    1. Loads the DINOv2 embeddings produced by :class:`ExtractEmbeddingsStep`.
    2. Computes each image's *mean cohesion*: its average cosine similarity to
       every other image in the place.
    3. Applies an iterative Tukey-fence outlier test (Q1 − ``tukey_k`` × IQR)
       to the cohesion scores.  The image with the lowest cohesion is removed
       if it falls below the fence; the test repeats until no further outliers
       are found.  No similarity threshold is required — the fence adapts to
       the distribution of similarities within each place.

    Chunk files are loaded on-demand and held in an LRU cache bounded by
    ``max_cached_chunks`` so memory stays manageable for million-image datasets.
    """

    def __init__(
        self,
        *,
        tukey_k: float = 1.5,
        max_cached_chunks: int = _MAX_CACHED_CHUNKS,
        context_key: str = "index",
        cache_dir_context_key: str = "embedding_cache_dir",
        place_id_column: str = "place_id",
        image_id_column: str = "image_id",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.tukey_k = tukey_k
        self.max_cached_chunks = max_cached_chunks
        self.context_key = context_key
        self.cache_dir_context_key = cache_dir_context_key
        self.place_id_column = place_id_column
        self.image_id_column = image_id_column

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        dataframe = context[self.context_key]
        cache_dir = Path(context[self.cache_dir_context_key])

        manifest = self._load_manifest(cache_dir)
        chunk_lru: OrderedDict[int, np.ndarray] = OrderedDict()

        keep_ids: list[str] = []
        place_groups = list(dataframe.groupby(self.place_id_column))

        with self.progress(total=len(place_groups), desc="filter visual overlap") as progress:
            for _place_id, group in place_groups:
                image_ids: list[str] = group[self.image_id_column].tolist()
                embeddings = self._load_embeddings(image_ids, manifest, cache_dir, chunk_lru)
                kept = self._filter_place(image_ids, embeddings)
                keep_ids.extend(kept)
                progress.update(1)

        keep_set = set(keep_ids)
        filtered = dataframe[dataframe[self.image_id_column].isin(keep_set)].reset_index(drop=True)

        context = dict(context)
        context[self.context_key] = filtered
        return context

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _load_manifest(self, cache_dir: Path) -> dict[str, tuple[int, int]]:
        """Return a dict mapping image_id -> (chunk_idx, row_idx)."""
        import pandas as pd

        manifest_path = cache_dir / "manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Embedding manifest not found at {manifest_path}. "
                "Run ExtractEmbeddingsStep first."
            )
        df = pd.read_parquet(manifest_path)
        return dict(
            zip(
                df["image_id"].tolist(),
                zip(df["chunk_idx"].tolist(), df["row_idx"].tolist()),
            )
        )

    # ------------------------------------------------------------------
    # Embedding lookup with LRU chunk cache
    # ------------------------------------------------------------------

    def _load_embeddings(
        self,
        image_ids: list[str],
        manifest: dict[str, tuple[int, int]],
        cache_dir: Path,
        chunk_lru: OrderedDict[int, np.ndarray],
    ) -> np.ndarray:
        rows: list[np.ndarray | None] = []
        for image_id in image_ids:
            entry = manifest.get(image_id)
            if entry is None:
                rows.append(None)
                continue
            chunk_idx, row_idx = entry
            chunk = self._get_chunk(chunk_idx, cache_dir, chunk_lru)
            rows.append(chunk[row_idx])

        # Determine embedding dim from first valid row
        embed_dim = next((r.shape[0] for r in rows if r is not None), 768)
        return np.stack(
            [r if r is not None else np.zeros(embed_dim, dtype=np.float32) for r in rows]
        )

    def _get_chunk(
        self,
        chunk_idx: int,
        cache_dir: Path,
        chunk_lru: OrderedDict[int, np.ndarray],
    ) -> np.ndarray:
        if chunk_idx in chunk_lru:
            # Move to end (most recently used)
            chunk_lru.move_to_end(chunk_idx)
            return chunk_lru[chunk_idx]

        chunk_path = cache_dir / "batches" / f"batch_{chunk_idx:06d}.npy"
        data = np.load(chunk_path)
        chunk_lru[chunk_idx] = data
        chunk_lru.move_to_end(chunk_idx)

        # Evict least-recently-used chunks when over the limit
        while len(chunk_lru) > self.max_cached_chunks:
            chunk_lru.popitem(last=False)

        return data

    # ------------------------------------------------------------------
    # Per-place filtering
    # ------------------------------------------------------------------

    def _filter_place(
        self, image_ids: list[str], embeddings: np.ndarray
    ) -> list[str]:
        """Return image_ids after iteratively removing low-cohesion outliers.

        Each iteration:
        - Compute every active image's mean cosine similarity to the other
          active images (its *cohesion*).
        - Compute Q1, Q3, IQR over those cohesion scores.
        - If the image with the lowest cohesion falls below Q1 − tukey_k×IQR,
          remove it and repeat; otherwise stop.

        Stops early when fewer than 3 images remain so we never over-prune a
        place into uselessness (the downstream RemoveSmallPlacesStep handles
        the minimum-size requirement).
        """
        n = len(image_ids)
        if n <= 2:
            return list(image_ids)

        # Pre-compute the full n×n cosine-similarity matrix once.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.maximum(norms, 1e-8)
        sim_full = normed @ normed.T  # [n, n], diagonal = 1.0

        active = list(range(n))

        while len(active) > 2:
            sub = sim_full[np.ix_(active, active)]
            # Mean similarity to *other* images (exclude self on the diagonal)
            np.fill_diagonal(sub, np.nan)
            cohesion = np.nanmean(sub, axis=1)  # shape [len(active)]

            q1, q3 = np.percentile(cohesion, [25, 75])
            iqr = q3 - q1
            lower_fence = q1 - self.tukey_k * iqr

            worst_local = int(np.argmin(cohesion))
            if cohesion[worst_local] < lower_fence:
                active.pop(worst_local)
            else:
                break

        return [image_ids[i] for i in active]
