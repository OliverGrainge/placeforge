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


@register_pipeline("pitts30k_val", category="val")
def build_pitts30k_pipeline() -> Pipeline:
    PIPELINE_NAME = "pitts30k_val"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / "pitts30k/images/val/queries",
                database_path=raw_dir() / "pitts30k/images/val/database",
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("tokyo247_val", category="val")
def build_tokyo247_pipeline() -> Pipeline:
    PIPELINE_NAME = "tokyo247_val"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / "tokyo247/images/test/queries",
                database_path=raw_dir() / "tokyo247/images/test/database",
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("sf_xl_small_val", category="val")
def build_tokyo247_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_small_val"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / "sf_xl/small/test/queries_v1/",
                database_path=raw_dir() / "sf_xl/small/test/database/",
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("sf_xl_val", category="val")
def build_tokyo247_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_val"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / "sf_xl/processed/val/queries/",
                database_path=raw_dir() / "sf_xl/processed/val/database/",
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )

