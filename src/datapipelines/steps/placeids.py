from __future__ import annotations

from math import floor
from typing import Any
import os

from pathlib import Path
import numpy as np
import pandas as pd

from .base import BaseStep
from .util import EmbeddingCache


class AssignPlaceIdStep(BaseStep):
    def __init__(self, cell_size_meters: float) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["dataset"].copy()
        cell_size = self.cell_size_meters

        df["cell_x"] = (df["utm_east"] / cell_size).apply(floor)
        df["cell_y"] = (df["utm_north"] / cell_size).apply(floor)
        df["place_id"] = df["cell_x"] * (df["cell_y"].max() + 1) + df["cell_y"]
        df = df.sort_values("place_id").reset_index(drop=True)

        return {**context, "dataset": df}


class AssignPlaceIdWithEmbedStep(BaseStep):
    """
    Iteratively remove incoherent images from each place.

    For each place:
      1. L2-normalise the image embeddings.
      2. Compute each image's mean cosine similarity to the other images.
      3. If the lowest mean similarity is below `cos_sim_threshold`, drop that
         image and repeat from step 2.
      4. Stop when every remaining image is above the threshold, or the place
         has been reduced to `min_place_size` images.

    Dropped images are removed from context["dataset"].
    """

    def __init__(
        self,
        embedding_name: str,
        cos_sim_threshold: float = 0.3,
        min_place_size: int = 2,
    ) -> None:
        super().__init__()
        self.cos_sim_threshold = cos_sim_threshold
        self.min_place_size = min_place_size
        feature_dir = Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"]) / embedding_name
        self.image_cache = EmbeddingCache(feature_dir / "images")

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["dataset"].copy()
        place_ids = df["place_id"].unique()

        if self.pbar is not None:
            self.pbar.reset(total=len(place_ids))

        keep_mask = np.ones(len(df), dtype=bool)

        # Pre-load the index once so we don't re-read parquet per place
        cache_index = self.image_cache.load_index().set_index("id")
        image_embs = self.image_cache.mmap()  # memory-mapped, ~0 RAM

        for place_id in place_ids:
            place_mask = df["place_id"] == place_id
            place_indices = df.index[place_mask].tolist()
            place_image_ids = df.loc[place_indices, "image_id"].tolist()

            if len(place_image_ids) <= self.min_place_size:
                if self.pbar is not None:
                    self.pbar.update(1)
                continue

            # Fetch embeddings for this place via the cache index
            rows = cache_index.loc[place_image_ids, "row"].values
            embeds = image_embs[rows].astype(np.float32)

            dropped = self._filter_place(embeds)

            # Mark dropped images in the global mask
            for local_idx in dropped:
                keep_mask[place_indices[local_idx]] = False

            if self.pbar is not None:
                self.pbar.update(1)

        df = df.loc[keep_mask].reset_index(drop=True)
        return {**context, "dataset": df}

    def _filter_place(self, embeds: np.ndarray) -> set[int]:
        """
        Returns the set of *local* indices (into `embeds`) to drop.
        """
        n = len(embeds)
        alive = list(range(n))  # local indices still in the running
        dropped: set[int] = set()

        # Normalise once — we'll re-slice as we remove outliers
        norms = np.linalg.norm(embeds, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # avoid division by zero
        normed = embeds / norms

        while len(alive) > self.min_place_size:
            subset = normed[alive]  # (K, D)
            # Pairwise cosine similarity (already L2-normalised)
            sim = subset @ subset.T  # (K, K)

            # Mean similarity excluding self (diagonal = 1.0)
            k = len(alive)
            np.fill_diagonal(sim, 0.0)
            mean_sim = sim.sum(axis=1) / (k - 1)

            worst = int(np.argmin(mean_sim))
            if mean_sim[worst] >= self.cos_sim_threshold:
                break  # all remaining images are coherent

            dropped.add(alive[worst])
            alive.pop(worst)

        return dropped

    







    