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





@register_pipeline("pitts30k")
def build_pitts30k_pipeline() -> Pipeline:
    PIPELINE_NAME = "pitts30k"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadValImagesStep(
                query_path=raw_dir() / "pitts30k/query", 
                database_path=raw_dir() / "pitts30k/database"
            ),
            ComputeValMatchesStep(radius_meters=25),
            SaveValDataset(name=PIPELINE_NAME),
            SummaryValDataset(name=PIPELINE_NAME),
        ],
    )