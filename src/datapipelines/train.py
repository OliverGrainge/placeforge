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
    SubsamplePlacesStep,
    SummaryTrainDataset,
    AnalyseSuperGroupStep,
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
                cell_size_meters=12.5,
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
                supergroup_size=1024,
                N=2,
                L=2,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
            AnalyseSuperGroupStep(
                name=name,
                place_embedding_name=name,
            ),
        ],
    )



# -----------------------------------------------------------------------------
# Combined: GSV-Cities + SF-XL + Pittsburgh 30k + MSLS
# -----------------------------------------------------------------------------

@register_pipeline("gsvcities_sf_xl_pitts30k_msls_train", category="train")
def build_gsvcities_sf_xl_pitts30k_msls_train() -> Pipeline:
    name = "gsvcities_sf_xl_pitts30k_msls_train"
    return Pipeline(
        name,
        steps=[
            # --- Read all four sources ---
            ReadGSVCitiesImagesStep(data_root=raw_dir() / GSVCITIES_PATH),
            ReadSFXLImagesStep(data_root=raw_dir() / SF_XL_PATH / "processed" / "train"),
            ReadPitts30kImagesStep(data_root=raw_dir() / PITTS30K_PATH / "images" / "train"),
            ReadMSLSImagesStep(data_root=raw_dir() / MSLS_PATH / "train"),
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
            ComputeImageEmbeddingStep(
                source="msls", batch_size=128, num_workers=8
            ),
            # --- Place-ID assignment + coherence filtering ---
            AssignCuraVPRPlaceIdStep(
                cell_size_meters=12.5,
                heading_size_degrees=60.0,
                cos_sim_threshold=0.3,
                min_images=4,
                use_heading=True,
            ),
            # --- Subsample non-GSV sources to balance the mix ---
            SubsamplePlacesStep(source="sf_xl", fraction=0.5),
            #SubsamplePlacesStep(source="pitts30k", fraction=0.5),
            SubsamplePlacesStep(source="msls", fraction=0.5),
            # --- Aggregate + supergroup ---
            AggregatePlaceEmbeddingStep(
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                supergroup_size=1024,
                N=2,
                L=2,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
            AnalyseSuperGroupStep(
                name=name,
                place_embedding_name=name,
            ),
        ],
    )

