from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
import pandas as pd
import os 
from PIL import Image


class VPRValidationDataset:
    def __init__(
        self,
        name: str,
        transform: Any = None,
    ) -> None:
        processed_dir = Path(os.environ["PLACEFORGE_PROCESSED_DIR"])
        self.dataset_dir = processed_dir / "val" / name
        self.parquet_path = self.dataset_dir / "valdataset.parquet"
        self.transform = transform

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Validation dataset not found: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)

        query_df = df[df["is_query"]].reset_index(drop=True)
        database_df = df[~df["is_query"]].reset_index(drop=True)

        self.database_records = self._build_database_records(database_df)
        self.query_records = self._build_query_records(query_df, database_offset=len(self.database_records))
        self.records = [*self.database_records, *self.query_records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        resolved_path = Path(os.environ["PLACEFORGE_PROCESSED_DIR"]) / record["image_path"]
        image = Image.open(resolved_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    @property
    def num_queries(self) -> int:
        return len(self.query_records)

    @property
    def num_database(self) -> int:
        return len(self.database_records)

    def ground_truth(self) -> list[tuple[int, list[int]]]:
        return self.query_df["matches"]
    


__all__ = ["VPRValidationDataset"]