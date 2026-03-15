from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.steps import (
    ReadTrainImagesStep,
    AssignPlaceIdStep,
    AssignSuperGroupStep,
    AssignPlaceIdWithEmbedStep,
    AssignSuperGroupWithEmbedStep,
    ComputeEmbeddingStep,
    SaveTrainDataset,
    SummaryTrainDataset,
)

EMBEDDING_NAME = "sf_xl_small"


@register_pipeline("sf_xl_small")
def build_test_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_small"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small"),
            AssignPlaceIdStep(cell_size_meters=10.0),
            AssignSuperGroupStep(total_supergroups=64, ),
            SaveTrainDataset(name=PIPELINE_NAME),
            SummaryTrainDataset(name=PIPELINE_NAME), 
        ],
    )


@register_pipeline("sf_xl_small_intra")
def build_test_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_small_intra"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small"),
            AssignPlaceIdStep(cell_size_meters=10.0),
            AssignSuperGroupStep(),
            ComputeEmbeddingStep(embedding_name=EMBEDDING_NAME, batch_size=64, num_workers=8),
            AssignPlaceIdWithEmbedStep(embedding_name=EMBEDDING_NAME, cos_sim_threshold=0.3, min_images=4),
            SaveTrainDataset(name=PIPELINE_NAME),
            SummaryTrainDataset(name=PIPELINE_NAME), 
        ],
    )


@register_pipeline("sf_xl_small_inter")
def build_test_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_small_inter"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small"),
            AssignPlaceIdStep(cell_size_meters=10.0),
            AssignSuperGroupStep(),
            ComputeEmbeddingStep(embedding_name=EMBEDDING_NAME, batch_size=64, num_workers=8),
            AssignSuperGroupWithEmbedStep(name=EMBEDDING_NAME, total_supergroups=64, kmeans_max_iter=100, seed=42),
            SaveTrainDataset(name=PIPELINE_NAME),
            SummaryTrainDataset(name=PIPELINE_NAME), 
        ],
    )


@register_pipeline("sf_xl_small_intra_inter")
def build_test_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_small_intra_inter"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small"),
            AssignPlaceIdStep(cell_size_meters=10.0),
            AssignSuperGroupStep(),
            ComputeEmbeddingStep(embedding_name=EMBEDDING_NAME, batch_size=64, num_workers=8),
            AssignPlaceIdWithEmbedStep(embedding_name=EMBEDDING_NAME, cos_sim_threshold=0.3, min_images=4),
            AssignSuperGroupWithEmbedStep(name=EMBEDDING_NAME, total_supergroups=64, kmeans_max_iter=100, seed=42),
            SaveTrainDataset(name=PIPELINE_NAME),
            SummaryTrainDataset(name=PIPELINE_NAME), 
        ],
    )