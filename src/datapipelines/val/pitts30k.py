from __future__ import annotations

from datapipelines import register_pipeline
from datapipelines.base import Pipeline
from datapipelines.env import processed_dir, raw_dir
from datapipelines.val.base import build_vpr_validation_pipeline


@register_pipeline("pitts30k")
def build_pitts30k_val() -> Pipeline:
    return build_vpr_validation_pipeline(
        pipeline_name="pitts30k",
        dataset_name="pitts30k",
        query_dir=raw_dir() / "pitts30k" / "images" / "val" / "queries",
        database_dir=raw_dir() / "pitts30k" / "images" / "val" / "database",
        output_dir=processed_dir() / "val" / "pitts30k",
        path_root=raw_dir(),
        match_radius_m=25.0,
    )
