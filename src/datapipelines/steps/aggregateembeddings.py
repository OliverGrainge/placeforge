from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseStep

_MAX_CACHED_CHUNKS = 256
_REDUCTIONS: dict[str, Any] = {
    "first": lambda values: values[0],
    "mean": lambda values: values.mean(axis=0),
    "median": lambda values: np.median(values, axis=0),
    "min": lambda values: values.min(axis=0),
    "max": lambda values: values.max(axis=0),
}


class AggregateEmbeddingsStep(BaseStep):
    """Aggregate extracted embeddings over a grouping key and cache the result.

    This step consumes the chunked cache produced by :class:`ExtractEmbeddingsStep`,
    groups rows in the dataset index by ``group_by_column``, reduces the member
    embeddings with ``reduction``, and writes the aggregated vectors back out in
    the same ``batches/*.npy`` + ``manifest.parquet`` layout.

    The resulting cache is resumable in the same way as the extraction cache:
    already-materialized groups are skipped on subsequent runs.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        group_by_column: str = "place_id",
        source_id_column: str = "image_id",
        reduction: str = "mean",
        batch_size: int = 256,
        normalize: bool = False,
        max_cached_chunks: int = _MAX_CACHED_CHUNKS,
        context_key: str = "index",
        source_cache_dir_context_key: str = "embedding_cache_dir",
        cache_dir_context_key: str | None = None,
        output_cache_dir_context_key: str = "aggregated_embedding_cache_dir",
        output_index_path_context_key: str = "aggregated_embedding_index_path",
        output_dim_context_key: str = "aggregated_embedding_dim",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if reduction not in _REDUCTIONS:
            raise ValueError(
                f"Unknown reduction {reduction!r}. Supported: {sorted(_REDUCTIONS)}"
            )

        self.cache_dir = Path(cache_dir)
        self.group_by_column = group_by_column
        self.source_id_column = source_id_column
        self.reduction = reduction
        self.batch_size = batch_size
        self.normalize = normalize
        self.max_cached_chunks = max_cached_chunks
        self.context_key = context_key
        self.source_cache_dir_context_key = source_cache_dir_context_key
        self.cache_dir_context_key = cache_dir_context_key
        self.output_cache_dir_context_key = output_cache_dir_context_key
        self.output_index_path_context_key = output_index_path_context_key
        self.output_dim_context_key = output_dim_context_key

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        dataframe = context[self.context_key]
        source_cache_dir = Path(context[self.source_cache_dir_context_key])
        cache_dir = self._resolve_cache_dir(context)

        source_manifest = self._load_source_manifest(source_cache_dir)
        embed_dim = self._resolve_embed_dim(context, source_cache_dir)
        cached_group_ids = self._load_cached_group_ids(cache_dir)
        total_groups = int(dataframe[self.group_by_column].nunique(dropna=False))
        completed_groups = self._count_completed_groups(dataframe, cached_group_ids)

        if completed_groups < total_groups:
            self._aggregate_and_cache(
                dataframe,
                source_manifest,
                source_cache_dir,
                cache_dir,
                cached_group_ids=cached_group_ids,
                embed_dim=embed_dim,
                total_groups=total_groups,
                completed_groups=completed_groups,
            )

        context = dict(context)
        context[self.output_cache_dir_context_key] = cache_dir
        context[self.output_index_path_context_key] = self._manifest_path(cache_dir)
        context[self.output_dim_context_key] = embed_dim
        return context

    def _resolve_cache_dir(self, context: dict[str, Any]) -> Path:
        if self.cache_dir_context_key and self.cache_dir_context_key in context:
            return Path(context[self.cache_dir_context_key])
        return self.cache_dir

    def _batches_dir(self, cache_dir: Path) -> Path:
        return cache_dir / "batches"

    def _manifest_path(self, cache_dir: Path) -> Path:
        return cache_dir / "manifest.parquet"

    def _load_source_manifest(self, cache_dir: Path) -> dict[Any, tuple[int, int]]:
        import pandas as pd

        manifest_path = cache_dir / "manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Embedding manifest not found at {manifest_path}. "
                "Run ExtractEmbeddingsStep first."
            )

        manifest = pd.read_parquet(manifest_path)
        return dict(
            zip(
                manifest[self.source_id_column].tolist(),
                zip(manifest["chunk_idx"].tolist(), manifest["row_idx"].tolist()),
            )
        )

    def _resolve_embed_dim(self, context: dict[str, Any], source_cache_dir: Path) -> int:
        context_dim = context.get("embedding_dim")
        if context_dim is not None:
            return int(context_dim)

        batch_paths = sorted(self._batches_dir(source_cache_dir).glob("batch_*.npy"))
        if not batch_paths:
            raise FileNotFoundError(
                f"No embedding batches found in {self._batches_dir(source_cache_dir)}"
            )

        sample = np.load(batch_paths[0], mmap_mode="r")
        return int(sample.shape[1])

    def _load_cached_group_ids(self, cache_dir: Path) -> set[Any]:
        manifest = self._load_output_manifest(cache_dir)
        if manifest is None:
            return set()
        return {self._cache_key(group_id) for group_id in manifest[self.group_by_column].tolist()}

    def _count_completed_groups(self, dataframe: Any, cached_group_ids: set[Any]) -> int:
        if not cached_group_ids:
            return 0

        unique_groups = dataframe[[self.group_by_column]].drop_duplicates()
        return sum(
            1
            for group_id in unique_groups[self.group_by_column].tolist()
            if self._cache_key(group_id) in cached_group_ids
        )

    def _cache_key(self, value: Any) -> Any:
        try:
            if np.isnan(value):
                return "__nan__"
        except TypeError:
            pass
        return value

    def _load_output_manifest(self, cache_dir: Path) -> Any:
        import pandas as pd

        manifest_path = self._manifest_path(cache_dir)
        if not manifest_path.exists():
            return None

        manifest = pd.read_parquet(manifest_path)
        if manifest.empty:
            return manifest

        valid_chunks = {
            int(path.stem.split("_")[1])
            for path in self._batches_dir(cache_dir).glob("batch_*.npy")
        }
        repaired = manifest[manifest["chunk_idx"].isin(valid_chunks)].reset_index(drop=True)
        if len(repaired) != len(manifest):
            self._write_manifest(cache_dir, repaired)
        return repaired

    def _next_chunk_idx(self, cache_dir: Path) -> int:
        max_idx = -1
        for path in self._batches_dir(cache_dir).glob("batch_*.npy"):
            try:
                chunk_idx = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            max_idx = max(max_idx, chunk_idx)
        return max_idx + 1

    def _aggregate_and_cache(
        self,
        dataframe: Any,
        source_manifest: dict[Any, tuple[int, int]],
        source_cache_dir: Path,
        cache_dir: Path,
        *,
        cached_group_ids: set[Any],
        embed_dim: int,
        total_groups: int,
        completed_groups: int,
    ) -> None:
        import pandas as pd

        self._batches_dir(cache_dir).mkdir(parents=True, exist_ok=True)
        chunk_lru: OrderedDict[int, np.ndarray] = OrderedDict()
        chunk_idx = self._next_chunk_idx(cache_dir)
        pending_group_ids: list[Any] = []
        pending_embeddings: list[np.ndarray] = []

        existing_manifest = self._load_output_manifest(cache_dir)
        accumulated_rows: list[dict[str, Any]] = (
            existing_manifest.to_dict("records") if existing_manifest is not None else []
        )

        with self.progress(
            total=total_groups,
            initial=completed_groups,
            desc="aggregate embeddings",
        ) as progress:
            for group_id, group in dataframe.groupby(self.group_by_column, sort=False, dropna=False):
                if self._cache_key(group_id) in cached_group_ids:
                    continue

                source_ids = group[self.source_id_column].tolist()
                source_embeddings = self._load_group_embeddings(
                    source_ids,
                    source_manifest,
                    source_cache_dir,
                    chunk_lru,
                )
                pending_group_ids.append(group_id)
                pending_embeddings.append(self._reduce(source_embeddings, embed_dim))

                if len(pending_group_ids) < self.batch_size:
                    continue

                rows = self._save_chunk(
                    cache_dir,
                    chunk_idx,
                    np.stack(pending_embeddings).astype(np.float32),
                    pending_group_ids,
                )
                accumulated_rows.extend(rows)
                self._write_manifest(cache_dir, pd.DataFrame(accumulated_rows))
                chunk_idx += 1
                progress.update(len(pending_group_ids))
                pending_group_ids = []
                pending_embeddings = []

            if pending_group_ids:
                rows = self._save_chunk(
                    cache_dir,
                    chunk_idx,
                    np.stack(pending_embeddings).astype(np.float32),
                    pending_group_ids,
                )
                accumulated_rows.extend(rows)
                self._write_manifest(cache_dir, pd.DataFrame(accumulated_rows))
                progress.update(len(pending_group_ids))

    def _load_group_embeddings(
        self,
        source_ids: list[Any],
        source_manifest: dict[Any, tuple[int, int]],
        source_cache_dir: Path,
        chunk_lru: OrderedDict[int, np.ndarray],
    ) -> np.ndarray:
        rows: list[np.ndarray] = []
        for source_id in source_ids:
            entry = source_manifest.get(source_id)
            if entry is None:
                continue
            chunk_idx, row_idx = entry
            rows.append(self._get_chunk(chunk_idx, source_cache_dir, chunk_lru)[row_idx])

        if not rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    def _get_chunk(
        self,
        chunk_idx: int,
        cache_dir: Path,
        chunk_lru: OrderedDict[int, np.ndarray],
    ) -> np.ndarray:
        if chunk_idx in chunk_lru:
            chunk_lru.move_to_end(chunk_idx)
            return chunk_lru[chunk_idx]

        chunk_path = self._batches_dir(cache_dir) / f"batch_{chunk_idx:06d}.npy"
        data = np.load(chunk_path)
        chunk_lru[chunk_idx] = data
        chunk_lru.move_to_end(chunk_idx)

        while len(chunk_lru) > self.max_cached_chunks:
            chunk_lru.popitem(last=False)

        return data

    def _reduce(self, values: np.ndarray, embed_dim: int) -> np.ndarray:
        if values.size == 0:
            reduced = np.zeros(embed_dim, dtype=np.float32)
        else:
            reduced = np.asarray(_REDUCTIONS[self.reduction](values), dtype=np.float32)

        if self.normalize:
            norm = np.linalg.norm(reduced)
            if norm > 0:
                reduced = reduced / norm

        return reduced.astype(np.float32, copy=False)

    def _save_chunk(
        self,
        cache_dir: Path,
        chunk_idx: int,
        embeddings: np.ndarray,
        group_ids: list[Any],
    ) -> list[dict[str, Any]]:
        chunk_path = self._batches_dir(cache_dir) / f"batch_{chunk_idx:06d}.npy"
        tmp_path = chunk_path.with_suffix(".tmp.npy")
        np.save(tmp_path, embeddings)
        tmp_path.replace(chunk_path)
        return [
            {self.group_by_column: group_id, "chunk_idx": chunk_idx, "row_idx": row_idx}
            for row_idx, group_id in enumerate(group_ids)
        ]

    def _flush_manifest(self, cache_dir: Path, new_rows: list[dict[str, Any]]) -> None:
        import pandas as pd

        new_df = pd.DataFrame(new_rows)
        manifest_path = self._manifest_path(cache_dir)
        if manifest_path.exists():
            existing = pd.read_parquet(manifest_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        self._write_manifest(cache_dir, new_df)

    def _write_manifest(self, cache_dir: Path, dataframe: Any) -> None:
        manifest_path = self._manifest_path(cache_dir)
        tmp_path = manifest_path.with_suffix(".tmp.parquet")
        dataframe.to_parquet(tmp_path, index=False)
        tmp_path.replace(manifest_path)
