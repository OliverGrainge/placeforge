from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import raw_dir
from datapipelines.paths import (
    NORDLAND_PATH,
    PITTS30K_PATH,
    SF_XL_PATH,
    SVOX_PATH,
    TOKYO247_PATH,
)
from datapipelines.steps import (
    ComputeTestMatchesStep,
    ReadTestImagesStep,
    SaveTestDataset,
    SummaryTestDataset,
)


@register_pipeline("pitts30k_test", category="test")
def build_pitts30k_pipeline() -> Pipeline:
    PIPELINE_NAME = "pitts30k_test"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTestImagesStep(
                query_path=raw_dir() / PITTS30K_PATH / "images" / "test" / "queries",
                database_path=raw_dir() / PITTS30K_PATH / "images" / "test" / "database",
            ),
            ComputeTestMatchesStep(radius_meters=25),
            SaveTestDataset(name=PIPELINE_NAME),
            SummaryTestDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("tokyo247_test", category="test")
def build_tokyo247_pipeline() -> Pipeline:
    PIPELINE_NAME = "tokyo247_test"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTestImagesStep(
                query_path=raw_dir() / TOKYO247_PATH / "images" / "test" / "queries",
                database_path=raw_dir() / TOKYO247_PATH / "images" / "test" / "database",
            ),
            ComputeTestMatchesStep(radius_meters=25),
            SaveTestDataset(name=PIPELINE_NAME),
            SummaryTestDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("sf_xl_small_test", category="test")
def build_sf_xl_small_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_small_test"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTestImagesStep(
                query_path=raw_dir() / SF_XL_PATH / "small" / "test" / "queries_v1",
                database_path=raw_dir() / SF_XL_PATH / "small" / "test" / "database",
            ),
            ComputeTestMatchesStep(radius_meters=25),
            SaveTestDataset(name=PIPELINE_NAME),
            SummaryTestDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("sf_xl_test", category="test")
def build_sf_xl_pipeline() -> Pipeline:
    PIPELINE_NAME = "sf_xl_test"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTestImagesStep(
                query_path=raw_dir() / SF_XL_PATH / "processed" / "test" / "queries_v1",
                database_path=raw_dir() / SF_XL_PATH / "processed" / "test" / "database",
            ),
            ComputeTestMatchesStep(radius_meters=25),
            SaveTestDataset(name=PIPELINE_NAME),
            SummaryTestDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("nordland_test", category="test")
def build_nordland_pipeline() -> Pipeline:
    PIPELINE_NAME = "nordland_test"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTestImagesStep(
                query_path=raw_dir() / NORDLAND_PATH / "images" / "test" / "queries",
                database_path=raw_dir() / NORDLAND_PATH / "images" / "test" / "database",
            ),
            ComputeTestMatchesStep(radius_meters=25),
            SaveTestDataset(name=PIPELINE_NAME),
            SummaryTestDataset(name=PIPELINE_NAME),
        ],
    )


@register_pipeline("svox_test", category="test")
def build_svox_pipeline() -> Pipeline:
    PIPELINE_NAME = "svox_test"
    return Pipeline(
        PIPELINE_NAME,
        steps=[
            ReadTestImagesStep(
                query_path=raw_dir() / SVOX_PATH / "images" / "test" / "queries",
                database_path=raw_dir() / SVOX_PATH / "images" / "test" / "gallery",
            ),
            ComputeTestMatchesStep(radius_meters=25),
            SaveTestDataset(name=PIPELINE_NAME),
            SummaryTestDataset(name=PIPELINE_NAME),
        ],
    )
