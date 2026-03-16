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
    ComputeImageEmbeddingStep,
    AggregatePlaceEmbeddingStep,
    SaveTrainDataset,
    SummaryTrainDataset,
)

# Shared image embedding cache (all pipelines using image embeddings share this)
IMAGE_EMBEDDING_NAME = "sf_xl_small"


@register_pipeline("sf_xl_small", category="train")
def build_sf_xl_small() -> Pipeline:
    name = "sf_xl_small"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small"),
            AssignPlaceIdStep(cell_size_meters=10.0),
            AssignSuperGroupStep(total_supergroups=64, adjacency_cells=2),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )


@register_pipeline("sf_xl_small_intra", category="train")
def build_sf_xl_small_intra() -> Pipeline:
    name = "sf_xl_small_intra"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small"),
            ComputeImageEmbeddingStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME, batch_size=64, num_workers=8
            ),
            AssignPlaceIdWithEmbedStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME,
                cell_size_meters=10.0,
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AssignSuperGroupStep(total_supergroups=64, adjacency_cells=2),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )


@register_pipeline("sf_xl_small_inter", category="train")
def build_sf_xl_small_inter() -> Pipeline:
    name = "sf_xl_small_inter"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small"),
            ComputeImageEmbeddingStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME, batch_size=64, num_workers=8
            ),
            AssignPlaceIdStep(cell_size_meters=10.0),
            AggregatePlaceEmbeddingStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME,
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignSuperGroupWithEmbedStep(
                place_embedding_name=name,
                total_supergroups=64,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )


@register_pipeline("sf_xl_small_intra_inter", category="train")
def build_sf_xl_small_intra_inter() -> Pipeline:
    name = "sf_xl_small_intra_inter"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small"),
            #AssignPlaceIdStep(cell_size_meters=10.0),
            ComputeImageEmbeddingStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME, batch_size=64, num_workers=8
            ),
            AssignPlaceIdWithEmbedStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME,
                cell_size_meters=10.0,
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AggregatePlaceEmbeddingStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME,
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignSuperGroupWithEmbedStep(
                place_embedding_name=name,
                total_supergroups=64,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )
