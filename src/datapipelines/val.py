from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.steps import (
    ReadValImagesStep,
    ComputeValMatchesStep,
    SaveValDataset,
    SummaryValDataset,
)

PITTS30K_PATH = "pitts30k"
TOKYO247_PATH = "tokyo247"
SF_XL_SMALL_PATH = "sf_xl"
SF_XL_PATH = "sf_xl"
MSLS_PATH = "msls"
NORDLAND_PATH = "VPR-datasets-downloader/datasets/nordland"
SVOX_PATH = "VPR-datasets-downloader/datasets/svox"


@register_pipeline("pitts30k_val", category="val")
def build_pitts30k_pipeline() -> Pipeline:
    PIPELINE_NAME = "pitts30k_val"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / PITTS30K_PATH / "images" / "val" / "queries",
                database_path=raw_dir() / PITTS30K_PATH / "images" / "val" / "database",
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )



@register_pipeline("sf_xl_small_val", category="val")
def build_sf_xl_small_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_small_val"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / SF_XL_SMALL_PATH / "small" / "val" / "queries",
                database_path=raw_dir() / SF_XL_SMALL_PATH / "small" / "val" / "database",
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("sf_xl_val", category="val")
def build_sf_xl_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_val"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / SF_XL_PATH / "processed" / "val" / "queries",
                database_path=raw_dir() / SF_XL_PATH / "processed" / "val" / "database",
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("msls_val", category="val")
def build_msls_pipeline() -> Pipeline:
    PIPELINE_NAME = "msls_val"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / MSLS_PATH / "val" / "queries",
                database_path=raw_dir() / MSLS_PATH / "val" / "database",
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("svox_val", category="val")
def build_svox_pipeline() -> Pipeline:
    PIPELINE_NAME = "svox_val"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / SVOX_PATH / "images" / "val" / "queries",
                database_path=raw_dir() / SVOX_PATH / "images" / "val" / "gallery",
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )