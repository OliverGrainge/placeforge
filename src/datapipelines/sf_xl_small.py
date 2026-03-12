from __future__ import annotations

from pathlib import Path

from . import register_pipeline
from .base import Pipeline
from .steps import (
    AssignPlaceIdsStep,
    AssignSuperGroupsStep,
    DatasetSummaryStep,
    ReadImagesStep,
    RemoveSmallPlacesStep,
    SaveFileStep,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SF_XL_SMALL_ROOT = Path("/mnt/datasets_ssd1/sf_xl/small")
SF_XL_SMALL_MATCH_RADIUS_METERS = 25.0
SF_XL_SMALL_OUTPUT_DIR = REPO_ROOT / "data" / "sf_xl_small_basic"
SF_XL_SMALL_INDEX_OUTPUT = SF_XL_SMALL_OUTPUT_DIR / "index.parquet"
SF_XL_SMALL_STATS_OUTPUT = SF_XL_SMALL_OUTPUT_DIR / "stats.json"


@register_pipeline("sf_xl_small")
def build_sf_xl_small_pipeline() -> Pipeline:
    return Pipeline(
        "sf_xl_small_index",
        steps=[
            # Scan the dataset directory and parse each image filename into a
            # row with UTM coordinates, GPS lat/lon, heading, and other metadata.
            ReadImagesStep(SF_XL_SMALL_ROOT),

            # Divide the UTM plane into a regular grid (cell side = match radius)
            # and assign each image a place_id for its cell.  Images sharing a
            # place_id are treated as positives during training. Images in
            # different place_ids have strictly non-overlapping image sets,
            # making them safe to treat as separate classes for metric learning.
            AssignPlaceIdsStep(cell_size_meters=SF_XL_SMALL_MATCH_RADIUS_METERS),

            # Drop undersized places so every surviving place_id has at least
            # this many images in the final index.
            RemoveSmallPlacesStep(min_images_per_place=4),

            # Colour the place_id grid with a 2×2 repeating pattern so that every
            # place_id within the same supergroup is at least 2 cells apart
            # (Chebyshev distance ≥ 2, never touching even at a corner).
            # During training, all place_ids sampled from the same supergroup are
            # guaranteed non-adjacent, so they can safely serve as hard negatives
            # within a single batch without risk of visual overlap.
            AssignSuperGroupsStep(cell_size_meters=SF_XL_SMALL_MATCH_RADIUS_METERS),

            # Persist the fully annotated index to parquet for use by the
            # training data loader.
            SaveFileStep(SF_XL_SMALL_INDEX_OUTPUT, context_key="index", output_path_context_key="index_path"),

            # Compute summary statistics over the index (place_id distribution,
            # supergroup distribution, etc.).
            DatasetSummaryStep(context_key="index", output_context_key="stats"),

            # Write the statistics to JSON alongside the index for reference.
            SaveFileStep(SF_XL_SMALL_STATS_OUTPUT, context_key="stats", output_path_context_key="stats_path"),
        ],
    )
