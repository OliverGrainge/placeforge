from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.paths import GSVCITIES_PATH, SF_XL_PATH, MSLS_PATH
from datapipelines.steps import (
    AggregatePlaceEmbeddingStep,
    AnalyseTrainDatasetStep,
    AssignCuraVPRPlaceIdStep,
    AssignCuraVPRSuperGroupStep,
    ComputeImageEmbeddingStep,
    ReadGSVCitiesTrainImagesStep,
    ReadTrainImagesStep,
    SaveTrainDataset,
    SummaryTrainDataset,
)

# -----------------------------------------------------------------------------
# SF_XL data pipelines (full dataset: processed/train)
# -----------------------------------------------------------------------------

@register_pipeline("sf_xl_train", category="train")
def build_sf_xl_train() -> Pipeline:
    name = "sf_xl_train"
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
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AggregatePlaceEmbeddingStep(
                image_embedding_name="sf_xl",
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=2048,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )


# -----------------------------------------------------------------------------
# Mixed SF_XL full + GSV-Cities data pipeline
# -----------------------------------------------------------------------------

@register_pipeline("sf_xl_gsvcities_train", category="train")
def build_sf_xl_gsvcities_cossim_train() -> Pipeline:
    name = "sf_xl_gsvcities_train"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(
                data_root=raw_dir() / SF_XL_PATH / "processed" / "train",
                source="sf_xl",
            ),
            ReadGSVCitiesTrainImagesStep(data_root=raw_dir() / GSVCITIES_PATH, source="gsvcities"),
            ComputeImageEmbeddingStep(
                image_embedding_name="sf_xl_gsvcities", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="sf_xl_gsvcities",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.45,
                min_images=6,
            ),
            AggregatePlaceEmbeddingStep(
                image_embedding_name="sf_xl_gsvcities",
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=1024,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )