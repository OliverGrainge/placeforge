from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


from .base import BaseStep


class SaveTrainDataset(BaseStep):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        processed_dir = os.environ["PLACEFORGE_PROCESSED_DIR"]
        output_dir = Path(processed_dir) / "train" / self.name
        output_path = output_dir / "traindataset.parquet"

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        df = context["traindataset"].copy()

        # Remap place_id to be sequential (0..n_places-1) within each supergroup_id
        place_df = df[["supergroup_id", "place_id"]].drop_duplicates("place_id")
        place_df = place_df.sort_values(["supergroup_id", "place_id"])
        place_df["local_place_id"] = place_df.groupby("supergroup_id").cumcount()
        old_to_local = place_df.set_index("place_id")["local_place_id"]
        df["place_id"] = df["place_id"].map(old_to_local).astype(np.int64)

        df.to_parquet(output_path)

        return context


class SaveValDataset(BaseStep):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        processed_dir = os.environ["PLACEFORGE_PROCESSED_DIR"]
        output_dir = Path(processed_dir) / "val" / self.name
        output_path = output_dir / "valdataset.parquet"

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        df = context["valdataset"]
        # Parquet can't serialize numpy arrays; convert matches to lists
        if "matches" in df.columns:
            df = df.copy()
            df["matches"] = df["matches"].apply(
                lambda m: m if isinstance(m, list) else []
            )
        df.to_parquet(output_path)

        return context


class SaveTestDataset(BaseStep):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        processed_dir = os.environ["PLACEFORGE_PROCESSED_DIR"]
        output_dir = Path(processed_dir) / "test" / self.name
        output_path = output_dir / "testdataset.parquet"

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        df = context["testdataset"]
        if "matches" in df.columns:
            df = df.copy()
            df["matches"] = df["matches"].apply(
                lambda m: m if isinstance(m, list) else []
            )
        df.to_parquet(output_path)

        return context
