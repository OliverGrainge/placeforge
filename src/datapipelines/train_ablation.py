from __future__ import annotations

from pathlib import Path

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.paths import SF_XL_PATH, GSVCITIES_PATH
from datapipelines.steps import (
    AggregatePlaceEmbeddingStep,
    AnalyseTrainDatasetStep,
    AssignCuraVPRPlaceIdStep,
    AssignCuraVPRSuperGroupStep,
    ComputeImageEmbeddingStep,
    ReadTrainImagesStep,
    SaveTrainDataset,
    SummaryTrainDataset,
    ReadGSVCitiesTrainImagesStep,
)

@register_pipeline("sf_xl_gsvcities_cossim_0p60_train", category="train")
def build_sf_xl_gsvcities_cossim_0p60_train() -> Pipeline:
    name = "sf_xl_gsvcities_cossim_0p60_train"
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
                cos_sim_threshold=0.6,
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
                supergroup_size=2048,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )

@register_pipeline("sf_xl_gsvcities_cossim_0p50_train", category="train")
def build_sf_xl_gsvcities_cossim_0p50_train() -> Pipeline:
    name = "sf_xl_gsvcities_cossim_0p50_train"
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
                cos_sim_threshold=0.5,
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
                supergroup_size=2048,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )


@register_pipeline("sf_xl_gsvcities_cossim_0p40_train", category="train")
def build_sf_xl_gsvcities_cossim_0p40_train() -> Pipeline:
    name = "sf_xl_gsvcities_cossim_0p40_train"
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
                cos_sim_threshold=0.4,
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
                supergroup_size=2048,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )

@register_pipeline("sf_xl_gsvcities_cossim_0p30_train", category="train")
def build_sf_xl_gsvcities_cossim_0p30_train() -> Pipeline:
    name = "sf_xl_gsvcities_cossim_0p30_train"
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
                supergroup_size=2048,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )




@register_pipeline("sf_xl_gsvcities_cossim_0p20_train", category="train")
def build_sf_xl_gsvcities_cossim_0p20_train() -> Pipeline:
    name = "sf_xl_gsvcities_cossim_0p20_train"
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
                cos_sim_threshold=0.2,
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
                supergroup_size=2048,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )



@register_pipeline("sf_xl_gsvcities_cossim_0p10_train", category="train")
def build_sf_xl_gsvcities_cossim_0p10_train() -> Pipeline:
    name = "sf_xl_gsvcities_cossim_0p10_train"
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
                cos_sim_threshold=0.1,
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
                supergroup_size=2048,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )