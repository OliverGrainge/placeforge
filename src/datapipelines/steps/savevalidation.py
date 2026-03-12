from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import BaseStep

_VPR_VAL_SCHEMA_VERSION = "1.0"


class SaveValidationSplitStep(BaseStep):
    """Save a query or database split to parquet.

    Reads the DataFrame at *context_key* and writes a parquet file with
    columns ``(<id_column>, path, lat, lon)`` using zstd compression.

    The *id_column* parameter selects which ID column to carry through
    (``"qid"`` for queries, ``"dbid"`` for database).  Image paths are stored
    relative to *path_root* when provided; otherwise the original paths are
    kept.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        id_column: str,
        context_key: str,
        path_root: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.output_path = Path(output_path)
        self.id_column = id_column
        self.context_key = context_key
        self.path_root = Path(path_root) if path_root is not None else None

    def run(self, context: dict[str, Any]) -> dict[str, Any] | None:
        if self.context_key not in context:
            raise KeyError(f"Context is missing dataframe at key {self.context_key!r}")

        import pandas as pd

        df = context[self.context_key]
        out = pd.DataFrame(
            {
                self.id_column: pd.array(df[self.id_column], dtype="uint32"),
                "path": df["image_path"].map(self._relativize),
                "lat": df["latitude"].astype("float64"),
                "lon": df["longitude"].astype("float64"),
            }
        )

        with self.progress(total=1, desc=f"save {self.context_key}") as progress:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(self.output_path, index=False, compression="zstd")
            progress.update(1)

        return None

    def _relativize(self, image_path: str) -> str:
        if self.path_root is None:
            return image_path
        return str(Path(image_path).relative_to(self.path_root))


class SaveMatchesStep(BaseStep):
    """Save the matches table to parquet.

    Reads the matches DataFrame at *context_key* and writes a parquet file
    with columns ``(qid, dbid)`` using zstd compression.  The file is written
    with a row group size of *row_group_size* (default 50 000) so that
    predicate pushdown on ``qid`` is efficient when reading subsets.

    The matches frame must already be sorted by ``qid`` (as produced by
    :class:`ComputeGeoMatchesStep`).
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        context_key: str = "matches",
        row_group_size: int = 50_000,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.output_path = Path(output_path)
        self.context_key = context_key
        self.row_group_size = row_group_size

    def run(self, context: dict[str, Any]) -> dict[str, Any] | None:
        if self.context_key not in context:
            raise KeyError(f"Context is missing dataframe at key {self.context_key!r}")

        import pandas as pd

        matches = context[self.context_key]
        out = pd.DataFrame(
            {
                "qid": pd.array(matches["qid"], dtype="uint32"),
                "dbid": pd.array(matches["dbid"], dtype="uint32"),
            }
        )

        with self.progress(total=1, desc="save matches") as progress:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(
                self.output_path,
                index=False,
                compression="zstd",
                engine="pyarrow",
                row_group_size=self.row_group_size,
            )
            progress.update(1)

        return None


class SaveValidationMetadataStep(BaseStep):
    """Write dataset-level metadata to ``metadata.json``.

    Records the dataset name, schema version, match radius, coordinate system,
    creation timestamp, and match statistics (total images, per-query match
    counts: mean, min, max, median).
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        dataset_name: str,
        match_radius_m: float,
        coordinate_system: str = "WGS84",
        queries_context_key: str = "queries",
        database_context_key: str = "database",
        matches_context_key: str = "matches",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.output_path = Path(output_path)
        self.dataset_name = dataset_name
        self.match_radius_m = match_radius_m
        self.coordinate_system = coordinate_system
        self.queries_context_key = queries_context_key
        self.database_context_key = database_context_key
        self.matches_context_key = matches_context_key

    def run(self, context: dict[str, Any]) -> dict[str, Any] | None:
        queries = context[self.queries_context_key]
        database = context[self.database_context_key]
        matches = context[self.matches_context_key]

        match_stats = self._match_stats(matches, n_queries=len(queries))

        meta = {
            "coordinate_system": self.coordinate_system,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": self.dataset_name,
            "match_radius_m": self.match_radius_m,
            "num_database_images": len(database),
            "num_query_images": len(queries),
            "num_total_images": len(queries) + len(database),
            "matches_per_query": match_stats,
            "version": _VPR_VAL_SCHEMA_VERSION,
        }

        with self.progress(total=1, desc="save metadata") as progress:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
            progress.update(1)

        return None

    @staticmethod
    def _match_stats(matches: Any, n_queries: int) -> dict[str, Any]:
        counts_series = matches.groupby("qid")["dbid"].count()
        # Include queries with zero matches in the distribution.
        all_counts = counts_series.reindex(range(n_queries), fill_value=0)
        return {
            "mean": round(float(all_counts.mean()), 3),
            "median": float(all_counts.median()),
            "min": int(all_counts.min()),
            "max": int(all_counts.max()),
            "num_queries_with_no_match": int((all_counts == 0).sum()),
        }
