from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.steps import (
    ReadTrainImagesStep,
    AssignPlaceIdStep,
    AssignSuperGroupStep,
    AssignPlaceIdWithEmbedStep,
    AssignCosPlacePlaceIdStep,
    AssignSuperGroupWithEmbedStep,
    ComputeImageEmbeddingStep,
    AssignCosPlaceSuperGroupStep,
    AggregatePlaceEmbeddingStep,
    AssignEigenPlacesPlaceIdStep,
    AssignEigenPlacesSuperGroupStep,
    AssignDiversePlaceIdWithEmbedStep,
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
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small/train/"),
            AssignPlaceIdStep(cell_size_meters=10.0),
            AssignSuperGroupStep(supergroup_size=64, adjacency_cells=2),
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
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small/train/"),
            ComputeImageEmbeddingStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME, batch_size=64, num_workers=8
            ),
            AssignPlaceIdWithEmbedStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME,
                cell_size_meters=10.0,
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AssignSuperGroupStep(supergroup_size=64, adjacency_cells=2),
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
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small/train/"),
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
                supergroup_size=64,
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
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small/train/"),
            # AssignPlaceIdStep(cell_size_meters=10.0),
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
                supergroup_size=64,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )


@register_pipeline("sf_xl_small_diverse_intra_inter", category="train")
def build_sf_xl_small_diverse_intra_inter() -> Pipeline:
    name = "sf_xl_small_diverse_intra_inter"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small/train/"),
            AssignPlaceIdStep(cell_size_meters=10.0),
            ComputeImageEmbeddingStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME, batch_size=64, num_workers=8
            ),
            AssignDiversePlaceIdWithEmbedStep(
                image_embedding_name=IMAGE_EMBEDDING_NAME,
                cos_sim_threshold=0.3,
                max_pair_similarity=0.94,
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
                supergroup_size=64,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )


@register_pipeline("sf_xl_small_cosplace", category="train")
def build_sf_xl_small_cosplace() -> Pipeline:
    name = "sf_xl_small_cosplace"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small/train/"),
            AssignCosPlacePlaceIdStep(cell_size_meters=10.0, heading_size_degrees=30.0),
            AssignCosPlaceSuperGroupStep(N=5, L=2),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )


@register_pipeline("sf_xl_small_eigenplaces", category="train")
def build_sf_xl_small_eigenplaces() -> Pipeline:
    name = "sf_xl_small_eigenplaces"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / "sf_xl/small/train/"),
            AssignEigenPlacesPlaceIdStep(
                cell_size_meters=15.0,
                focal_distance=10.0,
                heading_tolerance=30.0,
                min_images_per_place=2,
            ),
            AssignEigenPlacesSuperGroupStep(N=3),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )
