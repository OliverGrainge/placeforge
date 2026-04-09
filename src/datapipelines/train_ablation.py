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

# ---------------------------------------------------------------------------
# Dataset configs: per-dataset defaults for read step, embedding source,
# place-id params, and supergroup params.
# ---------------------------------------------------------------------------

_DATASET_CONFIGS = {
    "gsvcities": {
        "read_step": lambda: ReadGSVCitiesImagesStep(data_root=raw_dir() / GSVCITIES_PATH),
        "source": "gsvcities",
        "placeid_kwargs": dict(
            cell_size_meters=10.0,
            heading_size_degrees=30.0,
            min_images=4,
            use_heading=False,
        ),
        "supergroup_kwargs": dict(
            supergroup_size=512,
            N=5,
            L=2,
            kmeans_max_iter=100,
            seed=42,
        ),
    },
    "msls": {
        "read_step": lambda: ReadMSLSImagesStep(
            data_root=[
                raw_dir() / MSLS_PATH / "train" / "database",
                raw_dir() / MSLS_PATH / "train" / "queries",
            ],
        ),
        "source": "msls",
        "placeid_kwargs": dict(
            cell_size_meters=15.0,
            heading_size_degrees=60.0,
            min_images=4,
            use_heading=True,
        ),
        "supergroup_kwargs": dict(
            supergroup_size=512,
            N=3,
            L=2,
            kmeans_max_iter=100,
            seed=42,
        ),
    },
    "sf_xl": {
        "read_step": lambda: ReadSFXLImagesStep(
            data_root=raw_dir() / SF_XL_PATH / "processed" / "train",
        ),
        "source": "sf_xl",
        "placeid_kwargs": dict(
            cell_size_meters=10.0,
            heading_size_degrees=60.0,
            min_images=4,
        ),
        "supergroup_kwargs": dict(
            supergroup_size=512,
            N=5,
            L=2,
            kmeans_max_iter=100,
            seed=42,
        ),
    },
}


def _build_pipeline(
    name: str,
    dataset: str,
    cos_sim_threshold: float,
) -> Pipeline:
    cfg = _DATASET_CONFIGS[dataset]

    placeid_kwargs = dict(cfg["placeid_kwargs"])
    placeid_kwargs["cos_sim_threshold"] = cos_sim_threshold

    return Pipeline(
        name,
        steps=[
            cfg["read_step"](),
            ComputeImageEmbeddingStep(
                source=cfg["source"], batch_size=128, num_workers=8
            ),
            AssignCuraVPRPlaceIdStep(**placeid_kwargs),
            AggregatePlaceEmbeddingStep(
                place_embedding_name=name,
                reduction="mean",
                normalize=True,
            ),
            AssignCuraVPRSuperGroupStep(
                place_embedding_name=name,
                **cfg["supergroup_kwargs"],
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )


# ---------------------------------------------------------------------------
# Cosine similarity threshold ablation (supergroup_size fixed at 512)
# ---------------------------------------------------------------------------

_COS_SIM_VALUES = [0.2, 0.3, 0.4]

for _dataset in ("gsvcities", "msls", "sf_xl"):
    for _cos in _COS_SIM_VALUES:
        _cos_tag = str(_cos).replace(".", "")
        _name = f"{_dataset}_train_cos{_cos_tag}"

        def _factory(d=_dataset, c=_cos, n=_name):
            @register_pipeline(n, category="train")
            def _build() -> Pipeline:
                return _build_pipeline(n, d, cos_sim_threshold=c)
            return _build

        _factory()


# ---------------------------------------------------------------------------
# Combined: GSV-Cities + SF-XL + Pittsburgh 30k (baseline)
# Original settings: cell_size=10m, N=5 → min separation = 50m
# ---------------------------------------------------------------------------


@register_pipeline("gsvcities_sf_xl_pitts30k_baseline_train", category="train")
def build_gsvcities_sf_xl_pitts30k_baseline_train() -> Pipeline:
    name = "gsvcities_sf_xl_pitts30k_baseline_train"
    return Pipeline(
        name,
        steps=[
            # --- Read all three sources ---
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


# ---------------------------------------------------------------------------
# Combined: GSV-Cities + SF-XL + Pittsburgh 30k (clique-inspired)
# Decision-boundary negatives: cell_size=12.5m, N=2 → min separation = 25m
# ---------------------------------------------------------------------------


@register_pipeline("gsvcities_sf_xl_pitts30k_clique_train", category="train")
def build_gsvcities_sf_xl_pitts30k_clique_train() -> Pipeline:
    name = "gsvcities_sf_xl_pitts30k_clique_train"
    return Pipeline(
        name,
        steps=[
            # --- Read all three sources ---
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
                cell_size_meters=12.5,
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
                N=2,
                L=2,
                kmeans_max_iter=100,
                seed=42,
            ),
            SaveTrainDataset(name=name),
            SummaryTrainDataset(name=name),
            AnalyseTrainDatasetStep(name=name),
        ],
    )
