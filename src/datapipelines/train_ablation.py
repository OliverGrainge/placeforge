from __future__ import annotations

from pathlib import Path

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.paths import MSLS_PATH, SF_XL_PATH
from datapipelines.steps import (
    AggregatePlaceEmbeddingStep,
    AnalyseTrainDatasetStep,
    AssignCuraVPRPlaceIdStep,
    AssignCuraVPRSuperGroupStep,
    ComputeImageEmbeddingStep,
    ReadTrainImagesStep,
    SaveTrainDataset,
    SummaryTrainDataset,
)
def _build_train_cos_sim_ablation_pipeline(
    *,
    name: str,
    data_root: Path,
    image_embedding_name: str,
    cos_sim_threshold: float,
    source: str | None = None,
) -> Pipeline:
    read_kwargs: dict[str, Path | str] = {"data_root": data_root}
    if source is not None:
        read_kwargs["source"] = source

    return Pipeline(
        name,
        steps=[
            ReadTrainImagesStep(**read_kwargs),
            ComputeImageEmbeddingStep(
                image_embedding_name=image_embedding_name,
                batch_size=128,
                num_workers=8,
            ),
            AssignCuraVPRPlaceIdStep(
                image_embedding_name=image_embedding_name,
                cell_size_meters=10.0,
                heading_size_degrees=30.0,
                cos_sim_threshold=cos_sim_threshold,
                min_images=4,
            ),
            AggregatePlaceEmbeddingStep(
                image_embedding_name=image_embedding_name,
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
# SF_XL small cos_sim_threshold ablations
# -----------------------------------------------------------------------------


@register_pipeline("sf_xl_small_cossim_0p20_train", category="train")
def build_sf_xl_small_cossim_0p20_train() -> Pipeline:
    name = "sf_xl_small_cossim_0p20_train"
    return _build_train_cos_sim_ablation_pipeline(
        name=name,
        data_root=raw_dir() / SF_XL_PATH / "small" / "train",
        image_embedding_name="sf_xl_small",
        cos_sim_threshold=0.20,
    )


@register_pipeline("sf_xl_small_cossim_0p00_train", category="train")
def build_sf_xl_small_cossim_0p00_train() -> Pipeline:
    name = "sf_xl_small_cossim_0p00_train"
    return _build_train_cos_sim_ablation_pipeline(
        name=name,
        data_root=raw_dir() / SF_XL_PATH / "small" / "train",
        image_embedding_name="sf_xl_small",
        cos_sim_threshold=0.00,
    )


@register_pipeline("sf_xl_small_cossim_0p40_train", category="train")
def build_sf_xl_small_cossim_0p40_train() -> Pipeline:
    name = "sf_xl_small_cossim_0p40_train"
    return _build_train_cos_sim_ablation_pipeline(
        name=name,
        data_root=raw_dir() / SF_XL_PATH / "small" / "train",
        image_embedding_name="sf_xl_small",
        cos_sim_threshold=0.40,
    )


# -----------------------------------------------------------------------------
# MSLS cos_sim_threshold ablations
# -----------------------------------------------------------------------------


@register_pipeline("msls_cossim_0p20_train", category="train")
def build_msls_cossim_0p20_train() -> Pipeline:
    name = "msls_cossim_0p20_train"
    return _build_train_cos_sim_ablation_pipeline(
        name=name,
        data_root=raw_dir() / MSLS_PATH / "train",
        image_embedding_name="msls",
        cos_sim_threshold=0.20,
        source="msls",
    )


@register_pipeline("msls_cossim_0p00_train", category="train")
def build_msls_cossim_0p00_train() -> Pipeline:
    name = "msls_cossim_0p00_train"
    return _build_train_cos_sim_ablation_pipeline(
        name=name,
        data_root=raw_dir() / MSLS_PATH / "train",
        image_embedding_name="msls",
        cos_sim_threshold=0.00,
        source="msls",
    )


@register_pipeline("msls_cossim_0p40_train", category="train")
def build_msls_cossim_0p40_train() -> Pipeline:
    name = "msls_cossim_0p40_train"
    return _build_train_cos_sim_ablation_pipeline(
        name=name,
        data_root=raw_dir() / MSLS_PATH / "train",
        image_embedding_name="msls",
        cos_sim_threshold=0.40,
        source="msls",
    )
