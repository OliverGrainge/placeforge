from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import BaseStep


class SummaryTrainDataset(BaseStep):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["dataset"]
        processed_dir = os.environ["PLACEFORGE_PROCESSED_DIR"]
        output_path = Path(processed_dir) / "train" / self.name / "summary.json"

        def place_stats(group_df):
            images_per_place = group_df.groupby("place_id").size()
            return {
                "num_images": len(group_df),
                "num_places": int(images_per_place.count()),
                "images_per_place": {
                    "mean": float(images_per_place.mean()),
                    "min": int(images_per_place.min()),
                    "max": int(images_per_place.max()),
                    "median": float(images_per_place.median()),
                },
            }

        summary = {
            "schema": {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "num_rows": len(df),
            },
            "num_images": len(df),
            "num_places": int(df["place_id"].nunique()),
            "num_supergroups": int(df["supergroup_id"].nunique()),
            "images_per_place": {
                "mean": float(df.groupby("place_id").size().mean()),
                "min": int(df.groupby("place_id").size().min()),
                "max": int(df.groupby("place_id").size().max()),
                "median": float(df.groupby("place_id").size().median()),
            },
            "supergroups": {
                "supergroup_" + str(supergroup_id): place_stats(group_df)
                for supergroup_id, group_df in df.groupby("supergroup_id")
            },
        }

        output_path.write_text(json.dumps(summary, indent=2))

        return context