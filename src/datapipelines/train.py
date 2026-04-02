from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.steps import (
    ReadTrainImagesStep,
    ReadGSVCitiesTrainImagesStep,
    AssignCuraVPRPlaceIdStep,
    AssignCuraVPRSuperGroupStep,
    ComputeImageEmbeddingStep,
    AggregatePlaceEmbeddingStep,
    SaveTrainDataset,
    SummaryTrainDataset,
    AnalyseTrainDatasetStep,
)




SF_XL_SMALL_PATH = "sf_xl"
SF_XL_PATH = "sf_xl"
GSVCITIES_PATH = "gsvcities"
MSLS_PATH = "msls"
NORDLAND_PATH = "nordland"
SVOX_PATH = "svox"
PITTS30K_PATH = "pitts30k"
TOKYO247_PATH = "tokyo247"

# -----------------------------------------------------------------------------
# SF_XL data pipelines (full dataset: processed/train)
# -----------------------------------------------------------------------------

@register_pipeline("sf_xl_curavpr", category="train")
def build_sf_xl_curavpr_con() -> Pipeline:
    name = "sf_xl_curavpr"
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

@register_pipeline("sf_xl_small_curavpr", category="train")
def build_sf_xl_small_curavpr_con() -> Pipeline:
    name = "sf_xl_small_curavpr"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / SF_XL_SMALL_PATH / "small" / "train"),
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
                supergroup_size=1024,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )



@register_pipeline("sf_xl_small_cosin_00", category="train")
def build_sf_xl_small_curavpr_con() -> Pipeline:
    name = "sf_xl_small_cosin_00"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / SF_XL_SMALL_PATH / "small" / "train"),
            ComputeImageEmbeddingStep(
                image_embedding_name="sf_xl_small", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="sf_xl_small",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.00,
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



@register_pipeline("sf_xl_small_cosin_15", category="train")
def build_sf_xl_small_curavpr_con() -> Pipeline:
    name = "sf_xl_small_cosin_00"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / SF_XL_SMALL_PATH / "small" / "train"),
            ComputeImageEmbeddingStep(
                image_embedding_name="sf_xl_small", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name="sf_xl_small",
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.15,
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



@register_pipeline("sf_xl_small_cosin_03", category="train")
def build_sf_xl_small_curavpr_con() -> Pipeline:
    name = "sf_xl_small_cosin_00"
    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(data_root=raw_dir() / SF_XL_SMALL_PATH / "small" / "train"),
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



@register_pipeline("gsvcities_curavpr", category="train")
def build_sf_xl_small_gsvcities_curavpr() -> Pipeline:
    name = "gsvcities_curavpr"
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
                cos_sim_threshold=0.1,
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
                N=1,    # default is 5
                L=1,    # default is 2
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )



