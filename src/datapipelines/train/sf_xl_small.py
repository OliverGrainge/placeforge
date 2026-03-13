from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import feature_store_dir, processed_dir, raw_dir
from datapipelines.steps import (
    AggregateEmbeddingsStep,
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
SF_XL_SMALL_MIN_IMAGES_PER_PLACE = 4
SF_XL_SMALL_SUPERGROUP_CELL_SIZE = 500.0
SF_XL_SMALL_SUPERGROUP_ADJACENCY_CELLS = 3
SF_XL_SMALL_EMBEDDING_MODEL = "dinov2_salad"
SF_XL_SMALL_EMBEDDING_BATCH_SIZE = 64
SF_XL_SMALL_EMBEDDING_IMAGE_SIZE = 322


def _sf_xl_small_raw_dir():
    return raw_dir() / "sf_xl" / "small"


def _sf_xl_small_embedding_cache_dir():
    return feature_store_dir() / "sf_xl_small" / "embedding_cache"


def _sf_xl_small_aggregated_embedding_cache_dir():
    return feature_store_dir() / "sf_xl_small" / "aggregated_embedding_cache"


def _dataset_outputs(output_dir):
    return [
        SaveFileStep(output_dir / "index.parquet"),
        DatasetSummaryStep(),
        SaveFileStep(output_dir / "stats.json", context_key="stats"),
    ]


@register_pipeline("sf_xl_small")
def build_sf_xl_small_pipeline() -> Pipeline:
    output_dir = processed_dir() / "train" / "sf_xl_small"
    return Pipeline(
        "sf_xl_small",
        steps=[
            ReadImagesStep(_sf_xl_small_raw_dir()),
            AssignPlaceIdsStep(cell_size_meters=SF_XL_SMALL_MATCH_RADIUS_METERS),
            RemoveSmallPlacesStep(min_images_per_place=SF_XL_SMALL_MIN_IMAGES_PER_PLACE),
            AssignSuperGroupsStep(
                cell_size_meters=SF_XL_SMALL_SUPERGROUP_CELL_SIZE,
                adjacency_cells=SF_XL_SMALL_SUPERGROUP_ADJACENCY_CELLS,
            ),
            *_dataset_outputs(output_dir),
        ],
    )


@register_pipeline("sf_xl_small_aggregated")
def build_sf_xl_small_aggregated_pipeline() -> Pipeline:
    output_dir = processed_dir() / "train" / "sf_xl_small_aggregated"
    return Pipeline(
        "sf_xl_small_aggregated",
        steps=[
            ReadImagesStep(_sf_xl_small_raw_dir()),
            AssignPlaceIdsStep(cell_size_meters=SF_XL_SMALL_MATCH_RADIUS_METERS),
            RemoveSmallPlacesStep(min_images_per_place=SF_XL_SMALL_MIN_IMAGES_PER_PLACE),
            AssignSuperGroupsStep(
                cell_size_meters=SF_XL_SMALL_SUPERGROUP_CELL_SIZE,
                adjacency_cells=SF_XL_SMALL_SUPERGROUP_ADJACENCY_CELLS,
            ),
            ExtractEmbeddingsStep(
                cache_dir=_sf_xl_small_embedding_cache_dir(),
                model_name=SF_XL_SMALL_EMBEDDING_MODEL,
                batch_size=SF_XL_SMALL_EMBEDDING_BATCH_SIZE,
                image_size=SF_XL_SMALL_EMBEDDING_IMAGE_SIZE,
            ),
            AggregateEmbeddingsStep(
                cache_dir=_sf_xl_small_aggregated_embedding_cache_dir(),
                group_by_column="place_id",
            ),
            *_dataset_outputs(output_dir),
        ],
    )


@register_pipeline("sf_xl_small_intraclass_pruned")
def build_sf_xl_small_intraclass_pruned_pipeline() -> Pipeline:
    output_dir = processed_dir() / "train" / "sf_xl_small_intraclass_pruned"
    return Pipeline(
        "sf_xl_small_intraclass_pruned",
        steps=[
            ReadImagesStep(_sf_xl_small_raw_dir()),
            AssignPlaceIdsStep(cell_size_meters=SF_XL_SMALL_MATCH_RADIUS_METERS),
            ExtractEmbeddingsStep(
                cache_dir=_sf_xl_small_embedding_cache_dir(),
                model_name=SF_XL_SMALL_EMBEDDING_MODEL,
                batch_size=SF_XL_SMALL_EMBEDDING_BATCH_SIZE,
                image_size=SF_XL_SMALL_EMBEDDING_IMAGE_SIZE,
            ),
            FilterVisualOverlapStep(min_remaining=SF_XL_SMALL_MIN_IMAGES_PER_PLACE),
            AssignSuperGroupsStep(
                cell_size_meters=SF_XL_SMALL_SUPERGROUP_CELL_SIZE,
                adjacency_cells=SF_XL_SMALL_SUPERGROUP_ADJACENCY_CELLS,
            ),
            *_dataset_outputs(output_dir),
        ],
    )

