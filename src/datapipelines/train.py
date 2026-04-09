from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.paths import GSVCITIES_PATH, PITTS30K_PATH, SF_XL_PATH, MSLS_PATH
from datapipelines.steps import (
    AggregatePlaceEmbeddingStep,
    AnalyseTrainDatasetStep,
    AssignCuraVPRPlaceIdStep,
    AssignCuraVPRSuperGroupStep,
    ComputeImageEmbeddingStep,
    ReadGSVCitiesImagesStep,
    ReadMSLSImagesStep,
    ReadPitts30kImagesStep,
    ReadSFXLImagesStep,
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
            ReadSFXLImagesStep(data_root=raw_dir() / SF_XL_PATH / "processed" / "train"),
            ComputeImageEmbeddingStep(
                source="sf_xl", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                cell_size_meters=10.0,
                heading_size_degrees=60.0,
                cos_sim_threshold=0.3,
                min_images=4,
            ),
            AggregatePlaceEmbeddingStep(
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=512,
                N=5,
                L=2,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )


# -----------------------------------------------------------------------------
# Pittsburgh 30k (train split: images/train/database)
# -----------------------------------------------------------------------------

@register_pipeline("pitts30k_train", category="train")
def build_pitts30k_train() -> Pipeline:
    name = "pitts30k_train"
    return Pipeline(
        name,
        steps=[
            ReadPitts30kImagesStep(
                data_root=raw_dir() / PITTS30K_PATH / "images" / "train",
            ),
            ComputeImageEmbeddingStep(
                source="pitts30k", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                cell_size_meters=15.0,
                heading_size_degrees=60.0,
                cos_sim_threshold=0.3,
                min_images=4,
                use_heading=True,
            ),
            AggregatePlaceEmbeddingStep(
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=512,
                N=1,
                L=1,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )


# -----------------------------------------------------------------------------
# MSLS (train split: train/database + train/queries, same layout as val)
# -----------------------------------------------------------------------------

@register_pipeline("msls_train", category="train")
def build_msls_train() -> Pipeline:
    name = "msls_train"
    return Pipeline(
        name,
        steps=[
            ReadMSLSImagesStep(
                data_root=[
                    raw_dir() / MSLS_PATH / "train" / "database",
                    raw_dir() / MSLS_PATH / "train" / "queries",
                ],
            ),
            ComputeImageEmbeddingStep(
                source="msls", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                cell_size_meters=15.0,
                heading_size_degrees=60.0,
                cos_sim_threshold=0.3,
                min_images=4,
                use_heading=True,
            ),
            AggregatePlaceEmbeddingStep(
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=512,
                N=3,
                L=2,
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
            ReadGSVCitiesImagesStep(data_root=raw_dir() / GSVCITIES_PATH),
            ComputeImageEmbeddingStep(
                source="gsvcities", batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=0.3,
                min_images=4,
                use_heading=False,
            ),
            AggregatePlaceEmbeddingStep(
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=512,
                N=5,
                L=2,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )


# -----------------------------------------------------------------------------
# Combined: GSV-Cities + SF-XL + Pittsburgh 30k
# -----------------------------------------------------------------------------

@register_pipeline("gsvcities_sf_xl_pitts30k_train", category="train")
def build_gsvcities_sf_xl_pitts30k_train() -> Pipeline:
    name = "gsvcities_sf_xl_pitts30k_train"
    return Pipeline(
        name,
        steps=[
            # --- Read all three sources (auto-merged via _merge_into_context) ---
            ReadGSVCitiesImagesStep(data_root=raw_dir() / GSVCITIES_PATH),
            ReadSFXLImagesStep(data_root=raw_dir() / SF_XL_PATH / "processed" / "train"),
            ReadPitts30kImagesStep(data_root=raw_dir() / PITTS30K_PATH / "images" / "train"),
            # --- Compute embeddings per source ---
            ComputeImageEmbeddingStep(
                source="gsvcities", batch_size=128, num_workers=8
            ),
            ComputeImageEmbeddingStep(
                source="sf_xl", batch_size=128, num_workers=8
            ),
            ComputeImageEmbeddingStep(
                source="pitts30k", batch_size=128, num_workers=8
            ),
            # --- Place-ID assignment + coherence filtering ---
            AssignCuraVPRPlaceIdStep(
                cell_size_meters=10.0,
                heading_size_degrees=60.0,
                cos_sim_threshold=0.3,
                min_images=4,
                use_heading=True,
            ),
            # --- Aggregate + supergroup ---
            AggregatePlaceEmbeddingStep(
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=1024,
                N=5,
                L=2,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )

