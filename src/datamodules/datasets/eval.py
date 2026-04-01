from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import torchvision.io
from torch.utils.data import Dataset

_CANDIDATES = [
    ("val", "valdataset.parquet"),
    ("test", "testdataset.parquet"),
]


class EvalDataset(Dataset):
    def __init__(
        self,
        name: str,
        transform: Any = None,
    ) -> None:
        processed_dir = Path(os.environ["PLACEFORGE_PROCESSED_DIR"])
        self.transform = transform

        for split, filename in _CANDIDATES:
            path = processed_dir / split / name / filename
            if path.exists():
                self.dataset_dir = path.parent
                self.parquet_path = path
                break
        else:
            searched = [
                str(processed_dir / split / name / filename)
                for split, filename in _CANDIDATES
            ]
            raise FileNotFoundError(
                f"Dataset '{name}' not found. Searched:\n  " + "\n  ".join(searched)
            )

        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
        self.df = pd.read_parquet(self.parquet_path).set_index("image_id")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.df.iloc[index]
        resolved_path = self.raw_dir / record["image_path"]
        image = torchvision.io.read_image(
            str(resolved_path), mode=torchvision.io.ImageReadMode.RGB
        )
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
