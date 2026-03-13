from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.steps import (
    ReadImagesStep, 
    AssignPlaceIdStep, 
    AssignSuperGroupStep, 
    SaveTrainDataset
)


@register_pipeline(name=)