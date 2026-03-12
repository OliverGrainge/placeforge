from __future__ import annotations

from . import register_pipeline
from .base import Pipeline
from .env import feature_store_dir, processed_dir, raw_dir
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

SF_XL_SMALL_MATCH_RADIUS_METERS = 25.0


@register_pipeline("sf_xl_small")
def build_sf_xl_small_pipeline() -> Pipeline:
    output_dir = processed_dir() / "train" / "sf_xl_small_basic"
    return Pipeline(
        "sf_xl_small_index",
        steps=[
            # Scan the dataset directory and parse each image filename into a
            # row with UTM coordinates, GPS lat/lon, heading, and other metadata.
            ReadImagesStep(raw_dir() / "sf_xl" / "small"),

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
            SaveFileStep(output_dir / "index.parquet", context_key="index", output_path_context_key="index_path"),

            # Compute summary statistics over the index (place_id distribution,
            # supergroup distribution, etc.).
            DatasetSummaryStep(context_key="index", output_context_key="stats"),

            # Write the statistics to JSON alongside the index for reference.
            SaveFileStep(output_dir / "stats.json", context_key="stats", output_path_context_key="stats_path"),
        ],
    )


@register_pipeline("sf_xl_small_visual")
def build_sf_xl_small_visual_pipeline() -> Pipeline:
    output_dir = processed_dir() / "train" / "sf_xl_small_visual"
    embedding_cache_dir = feature_store_dir() / "sf_xl_small" / "embedding_cache"
    return Pipeline(
        "sf_xl_small_visual_index",
        steps=[
            # Scan the dataset directory and parse each image filename into a
            # row with UTM coordinates, GPS lat/lon, heading, and other metadata.
            ReadImagesStep(raw_dir() / "sf_xl" / "small"),

            # Divide the UTM plane into a regular grid (cell side = match radius)
            # and assign each image a place_id for its cell.
            AssignPlaceIdsStep(cell_size_meters=SF_XL_SMALL_MATCH_RADIUS_METERS),

            # Extract DINOv2 embeddings for every image and cache them to disk.
            # Already-cached images are skipped, so this step is fully resumable.
            # Embeddings are stored as chunked .npy files under the cache dir with
            # a manifest.parquet index. Provide embedding_cache_dir in the
            # initial pipeline context to override the default cache location.
            ExtractEmbeddingsStep(
                cache_dir=embedding_cache_dir,
                model_name="dinov2_vitb14_reg",
                batch_size=64,
                image_size=224,
                cache_dir_context_key="embedding_cache_dir",
                name="sf_xl_small_visual_dinov2_vitb14_reg"
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
                output_dir / "index.parquet",
                context_key="index",
                output_path_context_key="index_path",
            ),

            # Compute summary statistics.
            DatasetSummaryStep(context_key="index", output_context_key="stats"),

            # Write statistics to JSON alongside the index.
            SaveFileStep(
                output_dir / "stats.json",
                context_key="stats",
                output_path_context_key="stats_path",
            ),
        ],
    )
