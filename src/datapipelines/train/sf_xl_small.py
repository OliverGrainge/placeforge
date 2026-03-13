from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.steps import (
    ReadImagesStep,
    AssignPlaceIdStep,
    AssignSuperGroupStep,
    SaveTrainDataset,
)


@register_pipeline("test")
def build_test_pipeline() -> Pipeline:
    PIPELINE_NAME = "test"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadImagesStep(data_root=raw_dir() / "sf_xl/small"),
            AssignPlaceIdStep(cell_size_meters=10.0),
            AssignSuperGroupStep(),
            SaveTrainDataset(name=PIPELINE_NAME),
        ],
    )