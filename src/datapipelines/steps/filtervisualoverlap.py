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
    """Remove visually inconsistent images within each place.

    For every ``place_id`` group this step:

    1. Loads the DINOv2 embeddings produced by :class:`ExtractEmbeddingsStep`.
    2. Computes each image's *cohesion*, defined as the mean cosine similarity
       between that image and all other images in the same place.
    3. Iteratively removes visual outliers: the image with the lowest cohesion
       is removed if its cohesion falls below ``keep_threshold``.  The loop
       repeats on the remaining images until no further outliers are detected.

    After pruning, places with too few remaining images are discarded.  This
    ensures that each retained place still contains enough visually consistent
    images to support downstream tasks such as contrastive or triplet training.

    Embedding vectors are stored in chunked ``.npy`` files produced by the
    embedding extraction step.  Chunk files are loaded on demand and cached
    using an LRU policy so memory usage remains bounded even for large
    datasets.

    Parameters
    ----------
    keep_threshold : float
        Absolute cosine similarity cutoff in [-1, 1].  An image is removed if
        its mean cosine similarity to all other active images in the place falls
        below this value.  Because cosine similarity is scale-invariant, this
        threshold is consistent across all places.  A value of 0.3 is a
        reasonable starting point; increase it to be stricter, decrease to be
        more permissive.

    min_remaining : int
        Minimum number of images that must remain in a place after pruning.
        Places with fewer than this many images after filtering are discarded.

    max_cached_chunks : int
        Maximum number of embedding chunk files kept in memory simultaneously.
        This bounds memory usage when working with large embedding caches.

    context_key : str
        Context key containing the dataset index dataframe.

    cache_dir_context_key : str
        Context key pointing to the embedding cache directory produced by
        :class:`ExtractEmbeddingsStep`.

    place_id_column : str
        Name of the dataframe column identifying place groups.

    image_id_column : str
        Name of the dataframe column identifying images.

    Notes
    -----
    Cohesion is defined as the mean cosine similarity between an image and all
    other images in the same place.  Images with cohesion below
    ``keep_threshold`` are treated as visual outliers and removed.

    The pruning procedure operates purely within each place; images are never
    compared across places.
    """

    def __init__(
        self,
        *,
        keep_threshold: float = 0.3,
        min_remaining: int = 5,
        max_cached_chunks: int = _MAX_CACHED_CHUNKS,
        context_key: str = "index",
        cache_dir_context_key: str = "embedding_cache_dir",
        place_id_column: str = "place_id",
        image_id_column: str = "image_id",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.keep_threshold = keep_threshold
        self.min_remaining = min_remaining
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
        n = len(image_ids)

        if n == 0:
            return []

        # Pre-compute cosine similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normed = embeddings / np.maximum(norms, 1e-8)
        sim_full = normed @ normed.T

        active = list(range(n))

        while len(active) > 1:
            sub = sim_full[np.ix_(active, active)].copy()
            np.fill_diagonal(sub, np.nan)

            cohesion = np.nanmean(sub, axis=1)

            if np.all(np.isnan(cohesion)):
                break

            worst_local = int(np.nanargmin(cohesion))

            if cohesion[worst_local] < self.keep_threshold:
                active.pop(worst_local)
            else:
                break  # All remaining images are coherent enough

        result = [image_ids[i] for i in active]

        # Discard the whole place if too few coherent images remain
        if len(result) < self.min_remaining:
            return []
        return result