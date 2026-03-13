from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.steps import (
    ReadImagesStep,
    AssignPlaceIdStep,
    AssignSuperGroupStep,
    ComputeEmbeddingStep,
    SaveTrainDataset,
    SummaryTrainDataset,
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
            ComputeEmbeddingStep(name=PIPELINE_NAME, batch_size=64, num_workers=8),
            SaveTrainDataset(name=PIPELINE_NAME),
            SummaryTrainDataset(name=PIPELINE_NAME),
            
        ],
    )