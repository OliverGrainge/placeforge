"""Place-ID assignment pipeline steps.

Each step inherits from ``BaseStep`` and is designed to be composed into a
training-data preprocessing pipeline via the ``context`` dictionary pattern.
"""

from __future__ import annotations

import os
from math import floor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .base import BaseStep
from .util import EmbeddingCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _image_cache(name: str) -> EmbeddingCache:
    feature_store = Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])
    return EmbeddingCache(feature_store / "embedding" / "image" / name)


def _validate_heading(df: pd.DataFrame) -> None:
    missing = df["heading"].isna().sum()
    if missing > 0:
        raise ValueError(
            f"use_heading=True but {missing} records are missing a heading value."
        )


def _assign_place_ids(
    df: pd.DataFrame,
    cell_size_meters: float,
    heading_size_degrees: float | None = None,
) -> pd.DataFrame:
    """Assign a ``place_id`` to each row based on a regular UTM grid.

    Parameters
    ----------
    df:
        DataFrame with ``utm_east``, ``utm_north``, and (optionally) ``heading``
        columns.
    cell_size_meters:
        Side length of each square grid cell in metres.
    heading_size_degrees:
        When provided, heading is quantised into buckets of this width and
        included as a third dimension of the place key.  ``cell_h`` is added
        to the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``cell_x``, ``cell_y``, (optionally ``cell_h``), and
        ``place_id`` columns added, sorted by ``place_id``.
    """
    df = df.copy()
    df["cell_x"] = (df["utm_east"] / cell_size_meters).apply(floor)
    df["cell_y"] = (df["utm_north"] / cell_size_meters).apply(floor)

    if heading_size_degrees is not None:
        df["cell_h"] = (df["heading"] / heading_size_degrees).apply(floor)
        n_y = df["cell_y"].max() + 1
        n_h = df["cell_h"].max() + 1
        df["place_id"] = df["cell_x"] * (n_y * n_h) + df["cell_y"] * n_h + df["cell_h"]
    else:
        df["place_id"] = df["cell_x"] * (df["cell_y"].max() + 1) + df["cell_y"]

    return df.sort_values("place_id").reset_index(drop=True)


def _load_image_embeddings(
    cache: EmbeddingCache, df: pd.DataFrame
) -> tuple[np.ndarray, pd.Series]:
    """Return the memory-mapped embedding matrix and a series mapping image
    IDs to their row indices in that matrix."""
    image_embs = cache.mmap()
    image_index = cache.load_index().set_index("id")["row"]
    return image_embs, image_index


def _build_groups(
    df: pd.DataFrame, image_index: pd.Series
) -> list[tuple[list[int], np.ndarray]]:
    """Group DataFrame rows by ``place_id`` and resolve their embedding rows."""
    return [
        (sub.index.tolist(), image_index.loc[sub["image_id"].values].values)
        for _, sub in df.groupby("place_id")
    ]


def _apply_keep_mask(df: pd.DataFrame, keep_mask: np.ndarray) -> pd.DataFrame:
    """Filter *df* by *keep_mask* and re-factorise ``place_id`` to be
    contiguous starting from 0."""
    df = df.loc[keep_mask].reset_index(drop=True)
    df["place_id"] = pd.factorize(df["place_id"], sort=True)[0]
    return df


def _normalize(embeddings: np.ndarray, device: torch.device) -> torch.Tensor:
    return F.normalize(
        torch.from_numpy(embeddings.astype(np.float32)).to(device), dim=-1
    )


# ---------------------------------------------------------------------------
# Step 1 – grid assignment + coherence filtering via embeddings
# ---------------------------------------------------------------------------


class AssignCuraVPRPlaceIdStep(BaseStep):
    """Assign place IDs on a UTM grid, then remove incoherent images.

    Within each place, images are iteratively removed if their mean cosine
    similarity to the other images in the place falls below
    ``cos_sim_threshold``.  Places that end up with fewer than
    ``min_images`` survivors are removed entirely.

    Parameters
    ----------
    image_embedding_name:
        Sub-directory name under ``$PLACEFORGE_FEATURE_STORE_DIR/embedding/image/``.
    cell_size_meters:
        Side length of the square spatial cells in metres.
    cos_sim_threshold:
        Minimum acceptable mean cosine similarity.  Images below this value
        are iteratively dropped.
    min_images:
        Places with fewer surviving images than this are discarded.
    """

    def __init__(
        self,
        image_embedding_name: str,
        cell_size_meters: float,
        cos_sim_threshold: float = 0.3,
        min_images: int = 2,
        use_heading: bool = True,
        heading_size_degrees: float = 30.0,
    ) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters
        self.cos_sim_threshold = cos_sim_threshold
        self.min_images = min_images
        self.use_heading = use_heading
        self.heading_size_degrees = heading_size_degrees
        self.image_cache = _image_cache(image_embedding_name)

    def cache_params(self) -> dict[str, Any]:
        return {
            "image_cache_dir": str(self.image_cache.cache_dir),
            "cell_size_meters": self.cell_size_meters,
            "cos_sim_threshold": self.cos_sim_threshold,
            "min_images": self.min_images,
            "use_heading": self.use_heading,
            "heading_size_degrees": self.heading_size_degrees,
        }

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        raw_df = context["traindataset"]
        if self.use_heading:
            _validate_heading(raw_df)
        df = _assign_place_ids(
            raw_df,
            self.cell_size_meters,
            heading_size_degrees=self.heading_size_degrees if self.use_heading else None,
        )
        image_embs, image_index = _load_image_embeddings(self.image_cache, df)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        groups = _build_groups(df, image_index)

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

        return {**context, "traindataset": _apply_keep_mask(df, keep_mask)}

    @torch.no_grad()
    def _filter_place(self, embeds: np.ndarray, device: torch.device) -> set[int]:
        """Return the set of *local* indices to drop from this place."""
        n = len(embeds)
        if n <= 1:
            return set()

        normed = _normalize(embeds, device)
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



