from __future__ import annotations

from pathlib import Path

from . import register_pipeline
from .base import Pipeline
from .steps import (
    AssignPlaceIdsStep,
    AssignSuperGroupsStep,
    DatasetSummaryStep,
    ExtractEmbeddingsStep,
    FilterVisualOverlapStep,
    ReadImagesStep,
    RemoveSmallPlacesStep,
    SaveFileStep,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SF_XL_SMALL_ROOT = Path("/mnt/datasets_ssd1/sf_xl/small")
SF_XL_SMALL_MATCH_RADIUS_METERS = 25.0
SF_XL_SMALL_VISUAL_OUTPUT_DIR = REPO_ROOT / "data" / "sf_xl_small_visual"
SF_XL_SMALL_VISUAL_EMBEDDING_CACHE_DIR = SF_XL_SMALL_VISUAL_OUTPUT_DIR / "embedding_cache"
SF_XL_SMALL_VISUAL_INDEX_OUTPUT = SF_XL_SMALL_VISUAL_OUTPUT_DIR / "index.parquet"
SF_XL_SMALL_VISUAL_STATS_OUTPUT = SF_XL_SMALL_VISUAL_OUTPUT_DIR / "stats.json"


@register_pipeline("sf_xl_small_visual")
def build_sf_xl_small_visual_pipeline() -> Pipeline:
    return Pipeline(
        "sf_xl_small_visual_index",
        steps=[
            # Scan the dataset directory and parse each image filename into a
            # row with UTM coordinates, GPS lat/lon, heading, and other metadata.
            ReadImagesStep(SF_XL_SMALL_ROOT),

            # Divide the UTM plane into a regular grid (cell side = match radius)
            # and assign each image a place_id for its cell.
            AssignPlaceIdsStep(cell_size_meters=SF_XL_SMALL_MATCH_RADIUS_METERS),

            # Extract DINOv2 embeddings for every image and cache them to disk.
            # Already-cached images are skipped, so this step is fully resumable.
            # Embeddings are stored as chunked .npy files under the cache dir with
            # a manifest.parquet index.
            ExtractEmbeddingsStep(
                cache_dir=SF_XL_SMALL_VISUAL_EMBEDDING_CACHE_DIR,
                model_name="dinov2_vitb14_reg",
                batch_size=64,
                image_size=224,
            ),

            # For each place, iteratively remove images whose mean cosine
            # similarity to the rest of the place falls below the Tukey lower
            # fence (Q1 − 1.5 × IQR).  This is a data-driven outlier test with
            # no fixed similarity threshold: it adapts to the spread of each
            # place's embedding distribution and removes only genuine visual
            # outliers (e.g. panoramas pointing the wrong direction or captured
            # at a different intersection than the majority).
            FilterVisualOverlapStep(),

            # Colour the place_id grid with a 2×2 repeating pattern so that
            # every place_id within the same supergroup is at least 2 cells
            # apart (Chebyshev distance ≥ 2), guaranteeing non-adjacent hard
            # negatives within a training batch.
            AssignSuperGroupsStep(cell_size_meters=SF_XL_SMALL_MATCH_RADIUS_METERS),

            # Drop undersized places as the very last processing step so the
            # minimum-image requirement is evaluated on the fully filtered index.
            RemoveSmallPlacesStep(min_images_per_place=4),

            # Persist the fully annotated index.
            SaveFileStep(
                SF_XL_SMALL_VISUAL_INDEX_OUTPUT,
                context_key="index",
                output_path_context_key="index_path",
            ),

            # Compute summary statistics.
            DatasetSummaryStep(context_key="index", output_context_key="stats"),

            # Write statistics to JSON alongside the index.
            SaveFileStep(
                SF_XL_SMALL_VISUAL_STATS_OUTPUT,
                context_key="stats",
                output_path_context_key="stats_path",
            ),
        ],
    )
