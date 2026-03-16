from __future__ import annotations

from math import floor
from typing import Any
import os

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .base import BaseStep
from .util import EmbeddingCache


class AssignPlaceIdStep(BaseStep):
    def __init__(self, cell_size_meters: float) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = _assign_place_ids(context["traindataset"], self.cell_size_meters)

        return {**context, "traindataset": df}


def _assign_place_ids(df: pd.DataFrame, cell_size_meters: float) -> pd.DataFrame:
    df = df.copy()
    df["cell_x"] = (df["utm_east"] / cell_size_meters).apply(floor)
    df["cell_y"] = (df["utm_north"] / cell_size_meters).apply(floor)
    df["place_id"] = df["cell_x"] * (df["cell_y"].max() + 1) + df["cell_y"]
    return df.sort_values("place_id").reset_index(drop=True)


class AssignPlaceIdWithEmbedStep(BaseStep):
    """
    Iteratively remove incoherent images from each place.

    For each place:
      1. L2-normalise the image embeddings.
      2. Compute each image's mean cosine similarity to the other images.
      3. If the lowest mean similarity is below `cos_sim_threshold`, drop that
         image and repeat from step 2.
      4. Stop when every remaining image is coherent.
      5. If the number of surviving images is below `min_images`, remove the
         entire place from the dataset.
    """

    def __init__(
        self,
        image_embedding_name: str,
        cell_size_meters: float,
        cos_sim_threshold: float = 0.3,
        min_images: int = 2,
    ) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters
        self.cos_sim_threshold = cos_sim_threshold
        self.min_images = min_images
        self.image_cache = EmbeddingCache(
            Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"]) / "embedding" / "image" / image_embedding_name
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = _assign_place_ids(context["traindataset"], self.cell_size_meters)

        image_embs = self.image_cache.mmap()
        image_index = self.image_cache.load_index().set_index("id")["row"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        groups = [
            (sub.index.tolist(), image_index.loc[sub["image_id"].values].values)
            for _, sub in df.groupby("place_id")
        ]

        if self.pbar is not None:
            self.pbar.reset(total=len(groups))

        keep_mask = np.ones(len(df), dtype=bool)

        for df_indices, image_rows in groups:
            dropped = self._filter_place(image_embs[image_rows], device)
            n_surviving = len(image_rows) - len(dropped)

            if n_surviving < self.min_images:
                for idx in df_indices:
                    keep_mask[idx] = False
            else:
                for local_idx in dropped:
                    keep_mask[df_indices[local_idx]] = False

            if self.pbar is not None:
                self.pbar.update(1)

        df = df.loc[keep_mask].reset_index(drop=True)
        df["place_id"] = pd.factorize(df["place_id"], sort=True)[0]
        return {**context, "traindataset": df}

    @torch.no_grad()
    def _filter_place(self, embeds: np.ndarray, device: torch.device) -> set[int]:
        n = len(embeds)
        if n <= 1:
            return set()

        normed = F.normalize(
            torch.from_numpy(embeds.astype(np.float32)).to(device), dim=-1
        )
        alive = list(range(n))
        dropped: set[int] = set()

        while len(alive) > 1:
            subset = normed[alive]
            sim = subset @ subset.T
            sim.fill_diagonal_(0.0)
            mean_sim = sim.sum(dim=1) / (len(alive) - 1)

            worst = int(mean_sim.argmin())
            if mean_sim[worst].item() >= self.cos_sim_threshold:
                break

            dropped.add(alive[worst])
            alive.pop(worst)

        return dropped
