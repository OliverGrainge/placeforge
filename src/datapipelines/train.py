from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.steps import (
    ReadTrainImagesStep,
    PrintTrainDataset,
    AssignCuraVPRPlaceIdStep,
    AssignCosPlacePlaceIdStep,
    AssignCuraVPRSuperGroupStep,
    ComputeImageEmbeddingStep,
    AssignCosPlaceSuperGroupStep,
    AggregatePlaceEmbeddingStep,
    AssignEigenPlacesPlaceIdStep,
    AssignEigenPlacesSuperGroupStep,
    SaveTrainDataset,
    SummaryTrainDataset,
)


SF_XL_PATH = "sf_xl"

# -----------------------------------------------------------------------------
# SF_XL data pipelines (full dataset: processed/train)
# -----------------------------------------------------------------------------

@register_pipeline("sf_xl_cosplace", category="train")
def build_sf_xl_cosplace() -> Pipeline:
    name = "sf_xl_cosplace"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / SF_XL_PATH / "processed" / "train"),
            AssignCosPlacePlaceIdStep(cell_size_meters=10.0, heading_size_degrees=30.0),
            AssignCosPlaceSuperGroupStep(N=5, L=2),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )


@register_pipeline("sf_xl_eigenplaces", category="train")
def build_sf_xl_eigenplaces() -> Pipeline:
    name = "sf_xl_eigenplaces"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / SF_XL_PATH / "processed" / "train"),
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



@register_pipeline("sf_xl_curavpr_con", category="train")
def build_sf_xl_curavpr_con() -> Pipeline:
    name = "sf_xl_curavpr_con"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / SF_XL_PATH / "processed" / "train"),
            ComputeImageEmbeddingStep(
                image_embedding_name="sf_xl", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="sf_xl",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.4,
                min_images=4,
            ),
            PrintTrainDataset(),
            AggregatePlaceEmbeddingStep(
                image_embedding_name="sf_xl",
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=512,
                kmeans_max_iter=100,
                seed=42,
            ),
            PrintTrainDataset(),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )



@register_pipeline("sf_xl_curavpr_cls", category="train")
def build_sf_xl_curavpr_cls() -> Pipeline:
    name = "sf_xl_curavpr_cls"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / SF_XL_PATH / "processed" / "train"),
            ComputeImageEmbeddingStep(
                image_embedding_name="sf_xl", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="sf_xl",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.4,
                min_images=4,
            ),
            PrintTrainDataset(),
            AggregatePlaceEmbeddingStep(
                image_embedding_name="sf_xl",
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=512,
                kmeans_max_iter=100,
                seed=42,
            ),
            PrintTrainDataset(),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
        ],
    )



