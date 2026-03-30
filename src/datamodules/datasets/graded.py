from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.io
from torch.utils.data import Dataset


class GradedSimilarityTrainDataset(Dataset):
    """Image-pair dataset for training with graded similarity labels.

    Each sample is a pair of images together with a continuous similarity
    score ψ ∈ [0, 1] (FoV overlap IoU).  Pairs are drawn from three pools
    following the batch composition from the GCL paper:

    * 50 % positive pairs        (ψ > 0.5)
    * 25 % soft-negative pairs   (0 < ψ ≤ 0.5)
    * 25 % hard-negative pairs   (ψ = 0, randomly sampled)
    """

    def __init__(
        self,
        name: str,
        transform: Any = None,
    ) -> None:
        processed_dir = Path(os.environ["PLACEFORGE_PROCESSED_DIR"])
        raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])

        dataset_dir = processed_dir / "train" / name
        df = pd.read_parquet(dataset_dir / "traindataset.parquet")
        pairs_df = pd.read_parquet(dataset_dir / "pairs.parquet")

        self._image_paths: list[str] = df["image_path"].tolist()
        self._raw_dir = raw_dir
        self.transform = transform
        self._n_images = len(df)

        # Split precomputed pairs into positive and soft-negative pools.
        pos_mask = pairs_df["similarity"] > 0.5
        self._positive_pairs = pairs_df[pos_mask][
            ["anchor_idx", "pair_idx", "similarity"]
        ].values.astype(np.float64)
        self._soft_negative_pairs = pairs_df[~pos_mask][
            ["anchor_idx", "pair_idx", "similarity"]
        ].values.astype(np.float64)

    def __len__(self) -> int:
        # Epoch size chosen so that each positive pair is seen ~once on average
        # (positives account for 50 % of samples).
        return max(1, 2 * len(self._positive_pairs))

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = random.random()
        if r < 0.5 and len(self._positive_pairs) > 0:
            row = self._positive_pairs[random.randint(0, len(self._positive_pairs) - 1)]
            anchor_idx, pair_idx, sim = int(row[0]), int(row[1]), float(row[2])
        elif r < 0.75 and len(self._soft_negative_pairs) > 0:
            row = self._soft_negative_pairs[random.randint(0, len(self._soft_negative_pairs) - 1)]
            anchor_idx, pair_idx, sim = int(row[0]), int(row[1]), float(row[2])
        else:
            anchor_idx = random.randint(0, self._n_images - 1)
            pair_idx = random.randint(0, self._n_images - 1)
            sim = 0.0

        anchor_img = self._load_image(anchor_idx)
        pair_img = self._load_image(pair_idx)

        return {
            "anchor": anchor_img,
            "pair": pair_img,
            "similarity": sim,
        }

    def _load_image(self, idx: int) -> torch.Tensor:
        img = torchvision.io.read_image(
            str(self._raw_dir / self._image_paths[idx]),
            mode=torchvision.io.ImageReadMode.RGB,
        )
        if self.transform is not None:
            img = self.transform(img)
        return img

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "anchors": torch.stack([s["anchor"] for s in batch]),
            "pairs": torch.stack([s["pair"] for s in batch]),
            "similarities": torch.tensor(
                [s["similarity"] for s in batch], dtype=torch.float32,
            ),
        }
