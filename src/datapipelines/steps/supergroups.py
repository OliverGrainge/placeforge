from __future__ import annotations

import os
import random as pyrandom
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .base import BaseStep
from .util import EmbeddingCache


class AssignSuperGroupStep(BaseStep):
    """Two-level supergroup assignment: spatial outer groups + random sub-assignment.

    1. **Outer group** — spatial modular arithmetic on (cell_x, cell_y) so
       no two places within ``adjacency_cells`` of each other share a group.
    2. **Sub-group** — within each outer group, places are randomly assigned
       to subclusters.  The number of subclusters per outer group is
       allocated proportionally so the total hits ``total_supergroups``.

    Parameters
    ----------
    adjacency_cells : int
        Minimum cell distance between places in the same outer group.
    total_supergroups : int
        Desired total number of supergroups across the dataset.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        adjacency_cells: int = 2,
        total_supergroups: int = 64,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.adjacency_cells = adjacency_cells
        self.total_supergroups = total_supergroups
        self.seed = seed

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()
        period = self.adjacency_cells + 1

        # --- 1. Outer group from spatial modular arithmetic -----------------
        place_df = (
            df[["place_id", "cell_x", "cell_y"]]
            .drop_duplicates("place_id")
            .reset_index(drop=True)
        )
        place_df["outer_group"] = (
            (place_df["cell_x"] % period) * period + (place_df["cell_y"] % period)
        )

        # --- 2. Allocate subclusters proportionally -------------------------
        outer_counts = place_df["outer_group"].value_counts()
        allocation = self._allocate_clusters(outer_counts, len(place_df))

        # --- 3. Random sub-assignment within each outer group ---------------
        rng = pyrandom.Random(self.seed)
        subgroup_col = np.zeros(len(place_df), dtype=np.int64)
        sg_offset = 0

        for og in place_df["outer_group"].unique():
            mask = place_df["outer_group"].values == og
            k = min(allocation[og], int(mask.sum()))
            indices = np.where(mask)[0]
            for idx in indices:
                subgroup_col[idx] = sg_offset + rng.randrange(k)
            sg_offset += k

        place_df["supergroup_id"] = subgroup_col

        # Map back to image-level dataframe
        place_to_sg = place_df.set_index("place_id")["supergroup_id"]
        df["supergroup_id"] = df["place_id"].map(place_to_sg).astype(np.int64)

        return {**context, "traindataset": df}

    def _allocate_clusters(
        self,
        outer_counts: pd.Series,
        total_places: int,
    ) -> dict[int, int]:
        """Distribute ``total_supergroups`` across outer groups proportionally,
        guaranteeing every non-empty group gets at least 1.
        Uses largest-remainder method so the sum is exactly total_supergroups.
        """
        n_outer = len(outer_counts)
        target = max(self.total_supergroups, n_outer)

        fracs = {og: (count / total_places) * target for og, count in outer_counts.items()}
        alloc = {og: max(1, int(f)) for og, f in fracs.items()}

        remaining = target - sum(alloc.values())
        if remaining > 0:
            remainders = {og: fracs[og] - alloc[og] for og in alloc}
            for og in sorted(remainders, key=remainders.get, reverse=True):
                if remaining <= 0:
                    break
                alloc[og] += 1
                remaining -= 1

        return alloc


class AssignSuperGroupWithEmbedStep(BaseStep):
    """
    Two-level supergroup assignment:

    1. **Outer group** — spatial modular arithmetic on (cell_x, cell_y) with
       adjacency=2, producing ``period² = 9`` outer groups.  No two places
       within 2 cells of each other share an outer group.

    2. **Sub-cluster** — within each outer group, run KMeans on L2-normalised
       place embeddings.  The number of subclusters per outer group is
       allocated proportionally to how many places it contains, so the
       total across all outer groups hits ``total_supergroups``.

    Parameters
    ----------
    name : str
        Embedding name (subfolder under PLACEFORGE_FEATURE_STORE_DIR).
    total_supergroups : int
        Desired total number of supergroups across the entire dataset.
    kmeans_max_iter : int
        Maximum KMeans iterations.
    seed : int
        Random seed for reproducible clustering.
    """

    ADJACENCY_CELLS = 2  # hardcoded spatial separation

    def __init__(
        self,
        embedding_name: str,
        total_supergroups: int = 64,
        kmeans_max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.total_supergroups = total_supergroups
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed
        feature_dir = Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"]) / embedding_name
        self.place_cache = EmbeddingCache(feature_dir / "places")

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()
        period = self.ADJACENCY_CELLS + 1  # 3

        # --- 1. Outer group from spatial modular arithmetic -----------------
        # --- 2. Load place embeddings ---------------------------------------
        cache_index = self.place_cache.load_index().set_index("id")
        place_embs = self.place_cache.mmap()

        place_df = (
            df[["place_id", "cell_x", "cell_y"]]
            .drop_duplicates("place_id")
            .copy()
        )
        place_df["outer_group"] = (
            (place_df["cell_x"] % period) * period + (place_df["cell_y"] % period)
        )

        # Map place_id → row in the mmap (but don't read the embeddings yet)
        place_df = place_df.reset_index(drop=True)
        place_df["emb_row"] = cache_index.loc[place_df["place_id"].values, "row"].values

        # --- 3. Allocate subclusters proportionally -------------------------
        outer_counts = place_df["outer_group"].value_counts()
        total_places = len(place_df)

        allocation = self._allocate_clusters(outer_counts, total_places)

        # --- 4. KMeans sub-clustering within each outer group ---------------
        subcluster_col = np.zeros(len(place_df), dtype=np.int64)
        unique_outer = place_df["outer_group"].unique()

        if self.pbar is not None:
            self.pbar.reset(total=len(unique_outer))

        sg_offset = 0  # running offset to produce globally unique IDs

        for og in unique_outer:
            mask = place_df["outer_group"].values == og
            # Only pull this outer group's embeddings from the mmap into RAM
            og_emb_rows = place_df.loc[mask, "emb_row"].values
            og_embeddings = place_embs[og_emb_rows].astype(np.float32)

            # L2-normalise for cosine-based clustering
            norms = np.linalg.norm(og_embeddings, axis=1, keepdims=True)
            og_embeddings = og_embeddings / np.maximum(norms, 1e-8)

            n_places = og_embeddings.shape[0]
            k = min(allocation[og], n_places)

            if k <= 1:
                subcluster_col[mask] = sg_offset
                sg_offset += 1
            else:
                km = KMeans(
                    n_clusters=k,
                    init="k-means++",
                    max_iter=self.kmeans_max_iter,
                    n_init=10,
                    random_state=self.seed,
                )
                subcluster_col[mask] = km.fit_predict(og_embeddings) + sg_offset
                sg_offset += k

            if self.pbar is not None:
                self.pbar.update(1)

        place_df["supergroup_id"] = subcluster_col
        place_df = place_df.drop(columns=["emb_row"])

        # Map back to the image-level dataframe
        place_to_sg = place_df.set_index("place_id")["supergroup_id"]
        df["supergroup_id"] = df["place_id"].map(place_to_sg).astype(np.int64)

        return {**context, "traindataset": df}

    def _allocate_clusters(
        self,
        outer_counts: pd.Series,
        total_places: int,
    ) -> dict[int, int]:
        """
        Distribute ``total_supergroups`` across outer groups proportionally
        to their place count, guaranteeing every non-empty group gets at least 1.

        Uses largest-remainder method so the sum is exactly total_supergroups.
        """
        n_outer = len(outer_counts)
        target = max(self.total_supergroups, n_outer)  # at least 1 per group

        # Fractional allocation
        fracs = {og: (count / total_places) * target for og, count in outer_counts.items()}

        # Floor allocation — everyone gets at least 1
        alloc = {og: max(1, int(f)) for og, f in fracs.items()}

        # Distribute remaining slots by largest fractional remainder
        remaining = target - sum(alloc.values())
        if remaining > 0:
            remainders = {og: fracs[og] - alloc[og] for og in alloc}
            for og in sorted(remainders, key=remainders.get, reverse=True):
                if remaining <= 0:
                    break
                alloc[og] += 1
                remaining -= 1

        return alloc