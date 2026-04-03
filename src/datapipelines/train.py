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
                supergroup_size=512,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )


# -----------------------------------------------------------------------------
# SF_XL small data pipelines (small dataset: small/train)
# -----------------------------------------------------------------------------

@register_pipeline("sf_xl_small_train", category="train")
def build_sf_xl_small_train() -> Pipeline:
    name = "sf_xl_small_train"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / SF_XL_PATH / "small" / "train"),
            ComputeImageEmbeddingStep(
                image_embedding_name="sf_xl_small", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="sf_xl_small",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AggregatePlaceEmbeddingStep(
                image_embedding_name="sf_xl_small",
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
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )




# -----------------------------------------------------------------------------
# MSLS data pipelines (dataset: train)
# -----------------------------------------------------------------------------

@register_pipeline("msls_train", category="train")
def build_msls_train() -> Pipeline:
    name = "msls_train"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / MSLS_PATH / "train", source="msls"),
            ComputeImageEmbeddingStep(
                image_embedding_name="msls", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="msls",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AggregatePlaceEmbeddingStep(
                image_embedding_name="msls",
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
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )


# -----------------------------------------------------------------------------
# GSV-Cities data pipeline
# -----------------------------------------------------------------------------

@register_pipeline("gsvcities_train", category="train")
def build_gsvcities_train() -> Pipeline:
    name = "gsvcities_train"
    return Pipeline(
        name,
        steps=[
            ReadGSVCitiesTrainImagesStep(data_root=raw_dir() / GSVCITIES_PATH, source="gsvcities"),
            ComputeImageEmbeddingStep(
                image_embedding_name="gsvcities", batch_size=32, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="gsvcities",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AggregatePlaceEmbeddingStep(
                image_embedding_name="gsvcities",
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=512,
                kmeans_max_iter=100,
                seed=42,
                N=1,  # default is 5
                L=1,  # default is 2
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )


# -----------------------------------------------------------------------------
# Mixed SF_XL small + GSV-Cities data pipeline
# -----------------------------------------------------------------------------

@register_pipeline("sf_xl_small_gsvcities_train", category="train")
def build_sf_xl_small_gsvcities_train() -> Pipeline:
    name = "sf_xl_small_gsvcities_train"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(
                data_root=raw_dir() / SF_XL_PATH / "small" / "train",
                source="sf_xl_small",
            ),
            ReadGSVCitiesTrainImagesStep(
                data_root=raw_dir() / GSVCITIES_PATH,
                source="gsvcities",
                cities=["PRG", "London", "Boston", "WashingtonDC"],
            ),
            ComputeImageEmbeddingStep(
                image_embedding_name="sf_xl_small_gsvcities", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="sf_xl_small_gsvcities",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AggregatePlaceEmbeddingStep(
                image_embedding_name="sf_xl_small_gsvcities",
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


# -----------------------------------------------------------------------------
# Mixed SF_XL full + GSV-Cities data pipeline
# -----------------------------------------------------------------------------

@register_pipeline("sf_xl_gsvcities_train", category="train")
def build_sf_xl_gsvcities_train() -> Pipeline:
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
                cos_sim_threshold=0.3,
                min_images=4,
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




# -----------------------------------------------------------------------------
# Mixed MSLS + GSV-Cities data pipeline
# -----------------------------------------------------------------------------

@register_pipeline("msls_gsvcities_train", category="train")
def build_msls_gsvcities_train() -> Pipeline:
    name = "msls_gsvcities_train"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(
                data_root=raw_dir() / MSLS_PATH / "train",
                source="msls",
            ),
            ReadGSVCitiesTrainImagesStep(data_root=raw_dir() / GSVCITIES_PATH, source="gsvcities"),
            ComputeImageEmbeddingStep(
                image_embedding_name="msls_gsvcities", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="msls_gsvcities",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AggregatePlaceEmbeddingStep(
                image_embedding_name="msls_gsvcities",
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
