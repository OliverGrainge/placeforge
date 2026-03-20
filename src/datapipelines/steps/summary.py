from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .base import BaseStep


class PrintTrainDataset(BaseStep):
    """Print basic statistics about the train dataset to stdout.

    Gracefully skips place-level and supergroup-level stats if the
    corresponding columns are absent from the DataFrame.
    """

    show_pbar = False

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"]
        print(" ")
        print(f"  num_images : {len(df)}")

        if "place_id" in df.columns:
            ipp = df.groupby("place_id").size()
            print(f"  num_places : {ipp.count()}")
            print(
                f"  images/place — avg: {ipp.mean():.1f}  "
                f"min: {ipp.min()}  max: {ipp.max()}"
            )

        if "supergroup_id" in df.columns:
            isg = df.groupby("supergroup_id").size()
            print(f"  num_supergroups : {isg.count()}")
            print(
                f"  images/supergroup — avg: {isg.mean():.1f}  "
                f"min: {isg.min()}  max: {isg.max()}"
            )

            if "place_id" in df.columns:
                ppsg = df.groupby("supergroup_id")["place_id"].nunique()
                print(
                    f"  places/supergroup — avg: {ppsg.mean():.1f}  "
                    f"min: {ppsg.min()}  max: {ppsg.max()}"
                )
        print(" ")

        return context


class SummaryTrainDataset(BaseStep):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"]
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


class SummaryValDataset(BaseStep):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["valdataset"]
        processed_dir = os.environ["PLACEFORGE_PROCESSED_DIR"]
        output_path = Path(processed_dir) / "val" / self.name / "summary.json"

        query_df = df[df["is_query"]]
        database_df = df[~df["is_query"]]

        match_counts = query_df["matches"].apply(
            lambda m: len(m) if m is not None else 0
        )
        queries_with_no_matches = int((match_counts == 0).sum())

        summary = {
            "schema": {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "num_rows": len(df),
            },
            "num_images": len(df),
            "num_query_images": len(query_df),
            "num_database_images": len(database_df),
            "matches_per_query": {
                "mean": float(match_counts.mean()),
                "min": int(match_counts.min()),
                "max": int(match_counts.max()),
                "median": float(match_counts.median()),
            },
            "queries_with_no_matches": queries_with_no_matches,
            "queries_with_no_matches_pct": (
                float(queries_with_no_matches / len(query_df) * 100)
                if len(query_df) > 0
                else 0.0
            ),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

        return context


class SummaryTestDataset(BaseStep):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["testdataset"]
        processed_dir = os.environ["PLACEFORGE_PROCESSED_DIR"]
        output_path = Path(processed_dir) / "test" / self.name / "summary.json"

        query_df = df[df["is_query"]]
        database_df = df[~df["is_query"]]

        match_counts = query_df["matches"].apply(
            lambda m: len(m) if m is not None else 0
        )
        queries_with_no_matches = int((match_counts == 0).sum())

        summary = {
            "schema": {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "num_rows": len(df),
            },
            "num_images": len(df),
            "num_query_images": len(query_df),
            "num_database_images": len(database_df),
            "matches_per_query": {
                "mean": float(match_counts.mean()),
                "min": int(match_counts.min()),
                "max": int(match_counts.max()),
                "median": float(match_counts.median()),
            },
            "queries_with_no_matches": queries_with_no_matches,
            "queries_with_no_matches_pct": (
                float(queries_with_no_matches / len(query_df) * 100)
                if len(query_df) > 0
                else 0.0
            ),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))

        return context
