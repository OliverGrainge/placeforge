from __future__ import annotations

import os
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


class AssignCuraVPRSuperGroupStep(BaseStep):
    """Two-level supergroup assignment using place embeddings.

    Supergroups partition the training set so that each group can be sampled
    independently during contrastive learning.  The two levels ensure that
    places in the same supergroup are both **spatially distant** (outer
    group) and **visually coherent** (sub-cluster).

    1. **Outer group** — modular arithmetic on (cell_x, cell_y) and,
       when available, cell_h.  Produces N*N (or N*N*L with heading)
       groups where no two places within N cells share a group.
    2. **Sub-cluster** — within each outer group, spherical KMeans on
       L2-normalised place embeddings splits places into clusters of
       approximately ``supergroup_size`` members.  Cluster counts are
       allocated proportionally to each outer group's population.

    Parameters
    ----------
    place_embedding_name : str
        Name of the place embedding cache (subfolder under
        ``$PLACEFORGE_FEATURE_STORE_DIR/embedding/place/``).
    supergroup_size : int
        Target number of places per supergroup.  The total number of
        supergroups is derived as ``max(1, total_places // supergroup_size)``.
    kmeans_max_iter : int
        Maximum spherical KMeans iterations.
    seed : int
        Random seed for reproducible clustering.
    N : int
        Spatial modulus — places in the same outer group are at least
        ``N`` cells apart in both x and y.
    L : int
        Heading modulus — places in the same outer group are at least
        ``L`` heading buckets apart.  Only used when ``cell_h`` is present.
    """

    def __init__(
        self,
        place_embedding_name: str,
        supergroup_size: int = 64,
        kmeans_max_iter: int = 100,
        seed: int = 42,
        N: int = 5,
        L: int = 2,
    ) -> None:
        super().__init__()
        self.supergroup_size = supergroup_size
        self.kmeans_max_iter = kmeans_max_iter
        self.seed = seed
        self.N = N
        self.L = L
        self.place_cache = EmbeddingCache(
            Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])
            / "embedding"
            / "place"
            / place_embedding_name
        )

    def cache_params(self) -> dict[str, Any]:
        return {
            "place_cache_dir": str(self.place_cache.cache_dir),
            "supergroup_size": self.supergroup_size,
            "kmeans_max_iter": self.kmeans_max_iter,
            "seed": self.seed,
            "N": self.N,
            "L": self.L,
        }

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()

        place_df = self._unique_places(df)
        place_df["outer_group"] = self._outer_groups(place_df)
        place_df["emb_row"] = self._resolve_embedding_rows(place_df["place_id"])
        place_df["supergroup_id"] = self._subcluster(place_df)

        place_to_sg = place_df.set_index("place_id")["supergroup_id"]
        df["supergroup_id"] = df["place_id"].map(place_to_sg).astype(np.int64)
        return {**context, "traindataset": df}

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _unique_places(df: pd.DataFrame) -> pd.DataFrame:
        cols = ["place_id", "cell_x", "cell_y"]
        if "cell_h" in df.columns:
            cols.append("cell_h")
        return df[cols].drop_duplicates("place_id").reset_index(drop=True)

    def _outer_groups(self, place_df: pd.DataFrame) -> pd.Series:
        x_mod = place_df["cell_x"] % self.N
        y_mod = place_df["cell_y"] % self.N
        if "cell_h" in place_df.columns:
            h_mod = place_df["cell_h"] % self.L
            return x_mod * (self.N * self.L) + y_mod * self.L + h_mod
        return x_mod * self.N + y_mod

    def _resolve_embedding_rows(self, place_ids: pd.Series) -> pd.Series:
        place_index = self.place_cache.load_index().set_index("id")["row"]
        place_index.index = place_index.index.astype(place_ids.dtype)
        rows = place_ids.map(place_index)
        n_missing = rows.isna().sum()
        if n_missing:
            raise ValueError(
                f"{n_missing} place_ids not found in place embedding index. "
                "Ensure AggregatePlaceEmbeddingStep ran on the same traindataset."
            )
        return rows.astype(np.int64)

    def _subcluster(self, place_df: pd.DataFrame) -> np.ndarray:
        place_embs = self.place_cache.mmap()
        total_k = max(1, len(place_df) // self.supergroup_size)
        og_counts = place_df["outer_group"].value_counts()

        labels = np.zeros(len(place_df), dtype=np.int64)
        unique_outer = place_df["outer_group"].unique()

        if self.pbar is not None:
            self.pbar.reset(total=len(unique_outer))

        sg_offset = 0
        for og in unique_outer:
            mask = place_df["outer_group"].values == og
            embs = place_embs[place_df.loc[mask, "emb_row"].values].astype(np.float32)
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.maximum(norms, 1e-8)

            k = max(1, round(og_counts[og] / len(place_df) * total_k))
            k = min(k, embs.shape[0])

            if k <= 1:
                labels[mask] = sg_offset
                sg_offset += 1
            else:
                labels[mask] = _spherical_kmeans(
                    embs, k=k, max_iter=self.kmeans_max_iter, seed=self.seed
                ) + sg_offset
                sg_offset += k

            if self.pbar is not None:
                self.pbar.update(1)

        return labels
