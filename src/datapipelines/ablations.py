"""Ablation data pipelines for SF-XL.

Generates dataset variants for three ablation studies:

A. Component ablation — isolates coherence filtering and supergroup construction.
B. Threshold sensitivity — varies the coherence cosine similarity threshold tau.
C. Supergroup size sensitivity — varies the number of places per supergroup.

The full CureVPR pipeline (``sf_xl_train``) serves as the shared default for
A4, B(tau=0.3), and C(sg=1024).
"""

from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.paths import SF_XL_PATH
from datapipelines.steps import (
    AggregatePlaceEmbeddingStep,
    AnalyseTrainDatasetStep,
    AssignCuraVPRPlaceIdStep,
    AssignCuraVPRSuperGroupStep,
    AssignUniformSuperGroupStep,
    ComputeImageEmbeddingStep,
    ReadSFXLImagesStep,
    SaveTrainDataset,
    SummaryTrainDataset,
)

# Shared constants
_DATA_ROOT = lambda: raw_dir() / SF_XL_PATH / "processed" / "train"
_CELL_SIZE = 12.5
_HEADING_DEG = 60.0
_MIN_IMAGES = 4
_DEFAULT_TAU = 0.3
_DEFAULT_SG_SIZE = 1024
_SG_N = 2
_SG_L = 2
_KMEANS_ITER = 100
_SEED = 42


def _common_prefix(name: str) -> list:
    """Read images + compute embeddings (shared by all variants)."""
    return [
        ReadSFXLImagesStep(data_root=_DATA_ROOT()),
        ComputeImageEmbeddingStep(source="sf_xl", batch_size=128, num_workers=8),
    ]


def _place_id_step(cos_sim_threshold: float) -> AssignCuraVPRPlaceIdStep:
    return AssignCuraVPRPlaceIdStep(
        cell_size_meters=_CELL_SIZE,
        heading_size_degrees=_HEADING_DEG,
        cos_sim_threshold=cos_sim_threshold,
        min_images=_MIN_IMAGES,
    )


def _supergroup_steps(name: str, supergroup_size: int = _DEFAULT_SG_SIZE) -> list:
    """Aggregate place embeddings + assign supergroups via spherical k-means."""
    return [
        AggregatePlaceEmbeddingStep(
            place_embedding_name=name,
            reduction="mean",
            normalize=True,
        ),
        AssignCuraVPRSuperGroupStep(
            place_embedding_name=name,
            supergroup_size=supergroup_size,
            N=_SG_N,
            L=_SG_L,
            kmeans_max_iter=_KMEANS_ITER,
            seed=_SEED,
        ),
    ]


def _suffix(name: str) -> list:
    """Save + summary (shared by all variants)."""
    return [
        SaveTrainDataset(name=name),
        SummaryTrainDataset(name=name),
        AnalyseTrainDatasetStep(name=name),
    ]


# ============================================================================
# A. Component ablation
# ============================================================================


@register_pipeline("sf_xl_abl_raw_gps", category="train")
def build_raw_gps() -> Pipeline:
    """No coherence filtering, no supergroups (raw GPS cells only)."""
    name = "sf_xl_abl_raw_gps"
    return Pipeline(
        name,
        steps=[
            *_common_prefix(name),
            _place_id_step(cos_sim_threshold=-1.0),
            AssignUniformSuperGroupStep(),
            *_suffix(name),
        ],
    )


@register_pipeline("sf_xl_abl_sg_only", category="train")
def build_sg_only() -> Pipeline:
    """No coherence filtering, with supergroups."""
    name = "sf_xl_abl_sg_only"
    return Pipeline(
        name,
        steps=[
            *_common_prefix(name),
            _place_id_step(cos_sim_threshold=-1.0),
            *_supergroup_steps(name),
            *_suffix(name),
        ],
    )


@register_pipeline("sf_xl_abl_filter_only", category="train")
def build_filter_only() -> Pipeline:
    """Coherence filtering, no supergroups."""
    name = "sf_xl_abl_filter_only"
    return Pipeline(
        name,
        steps=[
            *_common_prefix(name),
            _place_id_step(cos_sim_threshold=_DEFAULT_TAU),
            AssignUniformSuperGroupStep(),
            *_suffix(name),
        ],
    )


# NOTE: The full pipeline (filtering + supergroups) is ``sf_xl_train``
# defined in train.py. No duplicate needed here.


# ============================================================================
# B. Threshold sensitivity (tau)
# ============================================================================


@register_pipeline("sf_xl_abl_tau02", category="train")
def build_tau02() -> Pipeline:
    """Permissive coherence threshold (tau=0.2)."""
    name = "sf_xl_abl_tau02"
    return Pipeline(
        name,
        steps=[
            *_common_prefix(name),
            _place_id_step(cos_sim_threshold=0.2),
            *_supergroup_steps(name),
            *_suffix(name),
        ],
    )


@register_pipeline("sf_xl_abl_tau04", category="train")
def build_tau04() -> Pipeline:
    """Strict coherence threshold (tau=0.4)."""
    name = "sf_xl_abl_tau04"
    return Pipeline(
        name,
        steps=[
            *_common_prefix(name),
            _place_id_step(cos_sim_threshold=0.4),
            *_supergroup_steps(name),
            *_suffix(name),
        ],
    )


# NOTE: tau=0.3 (default) is ``sf_xl_train`` in train.py.


# ============================================================================
# C. Supergroup size sensitivity
# ============================================================================


@register_pipeline("sf_xl_abl_sg512", category="train")
def build_sg512() -> Pipeline:
    """Smaller supergroups (512 places each)."""
    name = "sf_xl_abl_sg512"
    return Pipeline(
        name,
        steps=[
            *_common_prefix(name),
            _place_id_step(cos_sim_threshold=_DEFAULT_TAU),
            *_supergroup_steps(name, supergroup_size=512),
            *_suffix(name),
        ],
    )


@register_pipeline("sf_xl_abl_sg2048", category="train")
def build_sg2048() -> Pipeline:
    """Larger supergroups (2048 places each)."""
    name = "sf_xl_abl_sg2048"
    return Pipeline(
        name,
        steps=[
            *_common_prefix(name),
            _place_id_step(cos_sim_threshold=_DEFAULT_TAU),
            *_supergroup_steps(name, supergroup_size=2048),
            *_suffix(name),
        ],
    )


# NOTE: supergroup_size=1024 (default) is ``sf_xl_train`` in train.py.
