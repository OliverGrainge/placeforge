from __future__ import annotations

import os
import random as pyrandom
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import faiss

from .base import BaseStep
from .util import EmbeddingCache

_USE_GPU = faiss.get_num_gpus() > 0


def _spherical_kmeans(X: np.ndarray, k: int, max_iter: int, seed: int) -> np.ndarray:
    """Spherical KMeans via faiss. Uses GPU automatically if faiss-gpu is installed."""
    n, d = X.shape
    km = faiss.Kmeans(d, k, niter=max_iter, seed=seed, spherical=True, gpu=_USE_GPU, verbose=False)
    km.train(X)
    _, labels = km.index.search(X, 1)
    return labels.ravel().astype(np.int64)


class AssignSuperGroupStep(BaseStep):
    """Two-level supergroup assignment: spatial outer groups + random sub-assignment.

    1. **Outer group** — spatial modular arithmetic on (cell_x, cell_y) so
       no two places within ``adjacency_cells`` of each other share a group.
    2. **Sub-group** — within each outer group, places are randomly assigned
       to subclusters.  The number of subclusters per outer group is
       allocated proportionally so the total approximates
       ``total_places / supergroup_size``.

    Parameters
    ----------
    adjacency_cells : int
        Minimum cell distance between places in the same outer group.
    supergroup_size : int
        Target number of places per supergroup.  The actual total number of
        supergroups is derived at runtime as
        ``max(1, total_places // supergroup_size)``.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        adjacency_cells: int = 2,
        supergroup_size: int = 64,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.adjacency_cells = adjacency_cells
        self.supergroup_size = supergroup_size
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
        place_df["outer_group"] = (place_df["cell_x"] % period) * period + (
            place_df["cell_y"] % period
        )

        total_places = len(place_df)
        total_supergroups = max(1, total_places // self.supergroup_size)

        # --- 2. Allocate subclusters proportionally -------------------------
        outer_counts = place_df["outer_group"].value_counts()
        allocation = self._allocate_clusters(
            outer_counts, total_places, total_supergroups
        )

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

    @staticmethod
    def _allocate_clusters(
        outer_counts: pd.Series,
        total_places: int,
        total_supergroups: int,
    ) -> dict[int, int]:
        """Distribute ``total_supergroups`` across outer groups proportionally,
        guaranteeing every non-empty group gets at least 1.
        Uses largest-remainder method so the sum is exactly total_supergroups.
        """
        n_outer = len(outer_counts)
        target = max(total_supergroups, n_outer)

        fracs = {
            og: (count / total_places) * target for og, count in outer_counts.items()
        }
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
    """Two-level supergroup assignment using place embeddings.

    1. **Outer group** — spatial modular arithmetic on (cell_x, cell_y) with
       adjacency=2, producing ``period² = 9`` outer groups.  No two places
       within 2 cells of each other share an outer group.

    2. **Sub-cluster** — within each outer group, run KMeans on L2-normalised
       place embeddings.  The number of subclusters per outer group is
       allocated proportionally to how many places it contains, so the
       total across all outer groups approximates
       ``total_places / supergroup_size``.

    Parameters
    ----------
    place_embedding_name : str
        Name of the place embedding cache (subfolder under PLACEFORGE_FEATURE_STORE_DIR).
    supergroup_size : int
        Target number of places per supergroup.  The actual total number of
        supergroups is derived at runtime as
        ``max(1, total_places // supergroup_size)``.
    kmeans_max_iter : int
        Maximum KMeans iterations.
    seed : int
        Random seed for reproducible clustering.
    """

    ADJACENCY_CELLS = 2

    def __init__(
        self,
        place_embedding_name: str,
        supergroup_size: int = 64,
        kmeans_max_iter: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.supergroup_size = supergroup_size
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed
        self.place_cache = EmbeddingCache(
            Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])
            / "embedding"
            / "place"
            / place_embedding_name
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()
        period = self.ADJACENCY_CELLS + 1

        # --- 1. Outer group from spatial modular arithmetic -----------------
        place_df = (
            df[["place_id", "cell_x", "cell_y"]]
            .drop_duplicates("place_id")
            .reset_index(drop=True)
        )
        place_df["outer_group"] = (place_df["cell_x"] % period) * period + (
            place_df["cell_y"] % period
        )

        total_places = len(place_df)
        total_supergroups = max(1, total_places // self.supergroup_size)

        # --- 2. Load place embeddings ---------------------------------------
        place_embs = self.place_cache.mmap()
        place_index = self.place_cache.load_index().set_index("id")["row"]
        # Align index dtype with place_id (parquet may use different int width)
        place_index.index = place_index.index.astype(place_df["place_id"].dtype)
        place_df["emb_row"] = place_df["place_id"].map(place_index)
        if place_df["emb_row"].isna().any():
            n_missing = place_df["emb_row"].isna().sum()
            raise ValueError(
                f"{n_missing} place_ids not found in place embedding index. "
                "Ensure AggregatePlaceEmbeddingStep ran on the same traindataset."
            )
        place_df["emb_row"] = place_df["emb_row"].astype(np.int64)

        # --- 3. Allocate subclusters proportionally -------------------------
        allocation = self._allocate_clusters(
            place_df["outer_group"].value_counts(), total_places, total_supergroups
        )

        # --- 4. KMeans sub-clustering within each outer group ---------------
        subcluster_col = np.zeros(len(place_df), dtype=np.int64)
        unique_outer = place_df["outer_group"].unique()

        if self.pbar is not None:
            self.pbar.reset(total=len(unique_outer))

        sg_offset = 0

        for og in unique_outer:
            mask = place_df["outer_group"].values == og
            og_embeddings = place_embs[place_df.loc[mask, "emb_row"].values].astype(
                np.float32
            )

            norms = np.linalg.norm(og_embeddings, axis=1, keepdims=True)
            og_embeddings = og_embeddings / np.maximum(norms, 1e-8)

            k = min(allocation[og], og_embeddings.shape[0])

            if k <= 1:
                subcluster_col[mask] = sg_offset
                sg_offset += 1
            else:
                labels = _spherical_kmeans(
                    og_embeddings, k=k, max_iter=self.kmeans_max_iter, seed=self.seed
                )
                subcluster_col[mask] = labels + sg_offset
                sg_offset += k

            if self.pbar is not None:
                self.pbar.update(1)

        place_df["supergroup_id"] = subcluster_col

        place_to_sg = place_df.set_index("place_id")["supergroup_id"]
        df["supergroup_id"] = df["place_id"].map(place_to_sg).astype(np.int64)

        return {**context, "traindataset": df}

    @staticmethod
    def _allocate_clusters(
        outer_counts: pd.Series,
        total_places: int,
        total_supergroups: int,
    ) -> dict[int, int]:
        """Distribute ``total_supergroups`` across outer groups proportionally,
        guaranteeing every non-empty group gets at least 1.
        Uses largest-remainder method so the sum is exactly total_supergroups.
        """
        n_outer = len(outer_counts)
        target = max(total_supergroups, n_outer)

        fracs = {
            og: (count / total_places) * target for og, count in outer_counts.items()
        }
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


class AssignCosPlaceSuperGroupStep(BaseStep):
    """Assign supergroup IDs following the CosPlace grouping strategy.

    Groups are formed by modular arithmetic on (cell_x, cell_y, cell_h),
    producing N * N * L total supergroups. By construction, no two places
    within the same group are spatially adjacent (they are at least
    M * (N-1) metres apart or α * (L-1) degrees apart).

    Assumes AssignCosPlacePlaceIdStep has already been run, so that
    cell_x, cell_y and cell_h columns are present.

    Parameters
    ----------
    N : int
        Spatial modulus — controls minimum cell separation within a group.
    L : int
        Heading modulus — controls minimum heading separation within a group.
    """

    def __init__(self, N: int = 5, L: int = 2) -> None:
        super().__init__()
        self.N = N
        self.L = L

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()

        u = df["cell_x"] % self.N
        v = df["cell_y"] % self.N
        w = df["cell_h"] % self.L

        df["supergroup_id"] = u * (self.N * self.L) + v * self.L + w

        return {**context, "traindataset": df}


class AssignEigenPlacesSuperGroupStep(BaseStep):
    """Assign supergroup IDs following the EigenPlaces partitioning strategy.

    The paper partitions cells into non-overlapping subsets by taking
    every N-th cell in both the x and y directions.  This guarantees
    that no two cells within the same subset are spatial neighbours,
    so they can safely be treated as independent classes within a
    single training epoch.

    The method produces N² supergroups via modular arithmetic on the
    cell coordinates:

        supergroup_id = (cell_x % N) * N + (cell_y % N)

    At training time one would iterate over supergroups (shifting
    after each epoch), using only the places inside each supergroup
    for a classification batch.

    Parameters
    ----------
    N : int
        Spacing factor — only 1 / N² of all cells appear in a single
        supergroup.  The paper uses N = 3 so that cells within the
        same supergroup are at least 2 × cell_size apart, ensuring
        no visual overlap.  This produces 9 supergroups.
    """

    def __init__(self, N: int = 3) -> None:
        super().__init__()
        self.N = N

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()

        if "cell_x" not in df.columns or "cell_y" not in df.columns:
            raise KeyError(
                "Columns 'cell_x' and 'cell_y' are required.  "
                "Run AssignEigenPlacesPlaceIdStep first."
            )

        # Modular arithmetic — mirrors the paper's epoch-shifting partition
        df["supergroup_id"] = (df["cell_x"] % self.N) * self.N + (df["cell_y"] % self.N)

        return {**context, "traindataset": df}