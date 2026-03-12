"""Shared factory for VPR validation pipelines.

Produces a self-contained validation dataset with the following layout::

    <output_dir>/
    ├── queries.parquet   – qid, path, lat, lon
    ├── database.parquet  – dbid, path, lat, lon
    ├── matches.parquet   – qid, dbid  (sorted by qid, zstd, ~50 k row groups)
    └── metadata.json     – name, version, match_radius_m, coordinate_system

Usage
-----
Register a concrete pipeline by calling :func:`build_vpr_validation_pipeline`
inside a ``@register_pipeline`` factory once the source paths are known::

    @register_pipeline("my_dataset_val")
    def build_my_dataset_val() -> Pipeline:
        return build_vpr_validation_pipeline(
            pipeline_name="my_dataset_val",
            dataset_name="My Dataset",
            query_dir=raw_dir() / "my_dataset" / "val" / "queries",
            database_dir=raw_dir() / "my_dataset" / "val" / "database",
            output_dir=processed_dir() / "val" / "my_dataset",
            path_root=raw_dir(),
            match_radius_m=25.0,
        )
"""
from __future__ import annotations

from pathlib import Path

from datapipelines.base import Pipeline
from datapipelines.steps import (
    AssignSplitIdsStep,
    ComputeGeoMatchesStep,
    ReadImagesStep,
    RemoveUnmatchedQueriesStep,
    SaveMatchesStep,
    SaveValidationMetadataStep,
    SaveValidationSplitStep,
)

DEFAULT_MATCH_RADIUS_M = 25.0


def build_vpr_validation_pipeline(
    *,
    pipeline_name: str,
    dataset_name: str,
    query_dir: str | Path,
    database_dir: str | Path,
    output_dir: str | Path,
    match_radius_m: float = DEFAULT_MATCH_RADIUS_M,
    path_root: str | Path | None = None,
) -> Pipeline:
    """Construct a VPR validation pipeline for any dataset.

    Parameters
    ----------
    pipeline_name:
        Internal name passed to :class:`~datapipelines.base.Pipeline`.
        Should end with ``_val`` by convention.
    dataset_name:
        Human-readable name written into ``metadata.json``.
    query_dir:
        Directory containing query images.
    database_dir:
        Directory containing database images.
    output_dir:
        Destination directory for the four output artefacts.
        Convention: ``processed_dir() / "val" / pipeline_name``.
        The pipeline name and directory name should always match.
    match_radius_m:
        Maximum Euclidean distance in metres (UTM) for a positive match.
    path_root:
        When provided, image paths in the parquet files are expressed
        relative to this directory.
    """
    output_dir = Path(output_dir)

    return Pipeline(
        pipeline_name,
        steps=[
            ReadImagesStep(query_dir, context_key="queries", name="read query images"),
            ReadImagesStep(database_dir, context_key="database", name="read database images"),
            AssignSplitIdsStep(),
            ComputeGeoMatchesStep(radius_meters=match_radius_m),
            RemoveUnmatchedQueriesStep(),
            SaveValidationSplitStep(
                output_dir / "queries.parquet",
                id_column="qid",
                context_key="queries",
                path_root=path_root,
            ),
            SaveValidationSplitStep(
                output_dir / "database.parquet",
                id_column="dbid",
                context_key="database",
                path_root=path_root,
            ),
            SaveMatchesStep(output_dir / "matches.parquet"),
            SaveValidationMetadataStep(
                output_dir / "metadata.json",
                dataset_name=dataset_name,
                match_radius_m=match_radius_m,
            ),
        ],
    )
