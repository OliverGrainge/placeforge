from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from .base import BaseStep


_Reduction = Literal["mean", "median"]


class AggregateEmbeddingsStep(BaseStep):
    """Aggregate per-image embeddings into one embedding per unique key value.

    Reads embeddings written by ``ExtractEmbeddingsStep`` and reduces all image
    embeddings sharing the same value of ``key`` (e.g. ``place_id``) into a
    single vector using the chosen ``reduction``.

    Results are persisted as a *feature store* under ``feature_store_dir``:

    .. code-block:: text

        feature_store_dir/
            embeddings.npy   # float32 array, shape (n_keys, embed_dim)
            ids.parquet      # single-column DataFrame: the ordered key values

    When ``recompute=False`` (default) and the feature store already exists the
    step loads from disk and skips aggregation entirely.

    The step adds two keys to the context:

    * ``output_embeddings_context_key`` – ``np.ndarray`` of shape
      ``(n_keys, embed_dim)``, one row per unique key value.
    * ``output_ids_context_key`` – ``list`` of key values whose order matches
      the rows of the embeddings array.
    """

    def __init__(
        self,
        feature_store_dir: str | Path,
        *,
        key: str = "place_id",
        reduction: _Reduction = "mean",
        recompute: bool = False,
        context_key: str = "index",
        image_id_column: str = "image_id",
        embedding_cache_dir_context_key: str = "embedding_cache_dir",
        embedding_dim_context_key: str = "embedding_dim",
        output_embeddings_context_key: str = "place_embeddings",
        output_ids_context_key: str = "place_embedding_ids",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if reduction not in ("mean", "median"):
            raise ValueError(f"reduction must be 'mean' or 'median', got {reduction!r}")
        self.feature_store_dir = Path(feature_store_dir)
        self.key = key
        self.reduction = reduction
        self.recompute = recompute
        self.context_key = context_key
        self.image_id_column = image_id_column
        self.embedding_cache_dir_context_key = embedding_cache_dir_context_key
        self.embedding_dim_context_key = embedding_dim_context_key
        self.output_embeddings_context_key = output_embeddings_context_key
        self.output_ids_context_key = output_ids_context_key

    # ------------------------------------------------------------------
    # Feature store paths
    # ------------------------------------------------------------------

    def _embeddings_path(self) -> Path:
        return self.feature_store_dir / "embeddings.npy"

    def _ids_path(self) -> Path:
        return self.feature_store_dir / "ids.parquet"

    def _store_exists(self) -> bool:
        return self._embeddings_path().exists() and self._ids_path().exists()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.recompute and self._store_exists():
            embeddings, key_ids = self._load_store()
        else:
            cache_dir = Path(context[self.embedding_cache_dir_context_key])
            embed_dim: int = context[self.embedding_dim_context_key]
            dataframe = context[self.context_key]
            embeddings, key_ids = self._aggregate(cache_dir, embed_dim, dataframe)
            self._save_store(embeddings, key_ids)

        context = dict(context)
        context[self.output_embeddings_context_key] = embeddings
        context[self.output_ids_context_key] = key_ids
        return context

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        cache_dir: Path,
        embed_dim: int,
        dataframe: Any,
    ) -> tuple[np.ndarray, list]:
        import pandas as pd

        manifest = pd.read_parquet(cache_dir / "manifest.parquet")

        index_cols = dataframe[[self.image_id_column, self.key]]
        merged = manifest.merge(index_cols, on=self.image_id_column, how="inner")

        key_ids_ordered = merged[self.key].unique().tolist()
        num_keys = len(key_ids_ordered)
        key_to_idx = {k: i for i, k in enumerate(key_ids_ordered)}

        batches_dir = cache_dir / "batches"
        chunks: dict[int, np.ndarray] = {}
        for chunk_idx in merged["chunk_idx"].unique():
            chunk_path = batches_dir / f"batch_{chunk_idx:06d}.npy"
            chunks[int(chunk_idx)] = np.load(chunk_path)

        key_rows: list[list[np.ndarray]] = [[] for _ in range(num_keys)]
        for _, row in merged.iterrows():
            embedding = chunks[int(row["chunk_idx"])][int(row["row_idx"])]
            key_rows[key_to_idx[row[self.key]]].append(embedding)

        reduce_fn = np.mean if self.reduction == "mean" else np.median
        embeddings = np.empty((num_keys, embed_dim), dtype=np.float32)

        with self.progress(
            total=num_keys,
            desc=f"aggregate embeddings by {self.key} ({self.reduction})",
        ) as progress:
            for i, rows in enumerate(key_rows):
                embeddings[i] = reduce_fn(np.stack(rows), axis=0) if rows else np.zeros(embed_dim, dtype=np.float32)
                progress.update(1)

        return embeddings, key_ids_ordered

    # ------------------------------------------------------------------
    # Feature store I/O
    # ------------------------------------------------------------------

    def _save_store(self, embeddings: np.ndarray, key_ids: list) -> None:
        import pandas as pd

        self.feature_store_dir.mkdir(parents=True, exist_ok=True)

        tmp_emb = self._embeddings_path().with_suffix(".tmp.npy")
        np.save(tmp_emb, embeddings)
        tmp_emb.replace(self._embeddings_path())

        tmp_ids = self._ids_path().with_suffix(".tmp.parquet")
        pd.DataFrame({self.key: key_ids}).to_parquet(tmp_ids, index=False)
        tmp_ids.replace(self._ids_path())

    def _load_store(self) -> tuple[np.ndarray, list]:
        import pandas as pd

        embeddings = np.load(self._embeddings_path())
        key_ids = pd.read_parquet(self._ids_path())[self.key].tolist()
        return embeddings, key_ids
