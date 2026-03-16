from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset


class ValDataset(Dataset):
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
            raise FileNotFoundError(
                f"Validation dataset not found: {self.parquet_path}"
            )

        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
        self.df = pd.read_parquet(self.parquet_path).set_index("image_id")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.df.iloc[index]
        resolved_path = self.raw_dir / record["image_path"]
        image = Image.open(resolved_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {"image_id": index, "image": image}

    @property
    def num_queries(self) -> int:
        return len(self.df[self.df["is_query"] == True])

    @property
    def num_database(self) -> int:
        return len(self.df[self.df["is_query"] == False])

    def ground_truth(self) -> list[tuple[int, list[int]]]:
        db_df = self.df[self.df["is_query"] == False]
        id_to_db_pos = {img_id: pos for pos, img_id in enumerate(db_df.index)}

        result = []
        for qid, row in self.df[self.df["is_query"] == True].iterrows():
            matches = row["matches"]
            if matches is None or len(matches) == 0:
                db_positions = []
            else:
                db_positions = [id_to_db_pos[m] for m in matches if m in id_to_db_pos]
            result.append((qid, db_positions))
        return result


__all__ = ["VPRValidationDataset"]
