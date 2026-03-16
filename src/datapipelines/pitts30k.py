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


@register_pipeline("pitts30k", category="val")
def build_pitts30k_pipeline() -> Pipeline:
    PIPELINE_NAME = "pitts30k"
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


@register_pipeline("tokyo247", category="val")
def build_tokyo247_pipeline() -> Pipeline:
    PIPELINE_NAME = "tokyo247"
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
