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
# Step 1 – plain grid assignment
# ---------------------------------------------------------------------------


class AssignPlaceIdStep(BaseStep):
    """Assign place IDs using a flat UTM grid (no embedding filtering)."""

    def __init__(
        self,
        cell_size_meters: float,
        use_heading: bool = False,
        heading_size_degrees: float = 30.0,
    ) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters
        self.use_heading = use_heading
        self.heading_size_degrees = heading_size_degrees

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"]
        if self.use_heading:
            _validate_heading(df)
        df = _assign_place_ids(
            df,
            self.cell_size_meters,
            heading_size_degrees=self.heading_size_degrees if self.use_heading else None,
        )
        return {**context, "traindataset": df}


# ---------------------------------------------------------------------------
# Step 2 – grid assignment + coherence filtering via embeddings
# ---------------------------------------------------------------------------


class AssignPlaceIdWithEmbedStep(BaseStep):
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
        use_heading: bool = False,
        heading_size_degrees: float = 30.0,
    ) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters
        self.cos_sim_threshold = cos_sim_threshold
        self.min_images = min_images
        self.use_heading = use_heading
        self.heading_size_degrees = heading_size_degrees
        self.image_cache = _image_cache(image_embedding_name)

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


# ---------------------------------------------------------------------------
# Step 3 – compass-rose (CoSPlace) grid assignment
# ---------------------------------------------------------------------------


class AssignCosPlacePlaceIdStep(BaseStep):
    """Assign place IDs on a UTM grid discretised by *heading* as well.

    Each (cell_x, cell_y, cell_h) triple becomes a distinct place, mirroring
    the CoSPlace training strategy.

    Parameters
    ----------
    cell_size_meters:
        Side length of the square spatial cells in metres.
    heading_size_degrees:
        Angular width of each heading bucket in degrees.
    """

    def __init__(
        self,
        cell_size_meters: float,
        heading_size_degrees: float,
    ) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters
        self.heading_size_degrees = heading_size_degrees

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()

        df = df.dropna(subset=["heading"]).reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "AssignCosPlacePlaceIdStep: no images have heading data. "
                "CoSPlace requires heading to be encoded in image filenames."
            )

        df["cell_x"] = (df["utm_east"] / self.cell_size_meters).apply(floor)
        df["cell_y"] = (df["utm_north"] / self.cell_size_meters).apply(floor)
        df["cell_h"] = (df["heading"] / self.heading_size_degrees).apply(floor)

        n_y = df["cell_y"].max() + 1
        n_h = df["cell_h"].max() + 1
        df["place_id"] = df["cell_x"] * (n_y * n_h) + df["cell_y"] * n_h + df["cell_h"]

        return {
            **context,
            "traindataset": df.sort_values("place_id").reset_index(drop=True),
        }


# ---------------------------------------------------------------------------
# Step 4 – EigenPlaces assignment
# ---------------------------------------------------------------------------


class AssignEigenPlacesPlaceIdStep(BaseStep):
    """Assign place IDs following the EigenPlaces training paradigm.

    For each spatial cell:

    1. Centre the UTM coordinates and compute their SVD.  The first principal
       component approximates the road direction; the second is perpendicular
       (towards the roadside).
    2. Place a **lateral** focal point at distance ``focal_distance`` along
       the second PC, and a **frontal** focal point along the first.
    3. Assign each image to a focal-point class when its heading is within
       ``heading_tolerance`` degrees of the angle towards that focal point.
    4. Discard (cell, class) pairs with fewer than ``min_images_per_place``
       survivors.

    Parameters
    ----------
    cell_size_meters:
        Side length of the square spatial cells in metres (paper default: 15 m).
    focal_distance:
        Distance *D* from the cell centroid to a focal point (paper default: 10 m).
    heading_tolerance:
        Maximum angular deviation (degrees) for an image to be included in a
        focal-point class.
    min_images_per_place:
        Minimum number of images required to retain a (cell, class) place.
    """

    def __init__(
        self,
        cell_size_meters: float = 15.0,
        focal_distance: float = 10.0,
        heading_tolerance: float = 30.0,
        min_images_per_place: int = 2,
    ) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters
        self.focal_distance = focal_distance
        self.heading_tolerance = heading_tolerance
        self.min_images_per_place = min_images_per_place

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()
        df = df.dropna(subset=["heading"]).reset_index(drop=True)

        df["cell_x"] = (df["utm_east"] / self.cell_size_meters).apply(floor)
        df["cell_y"] = (df["utm_north"] / self.cell_size_meters).apply(floor)

        groups = list(df.groupby(["cell_x", "cell_y"]))
        if self.pbar is not None:
            self.pbar.reset(total=len(groups))

        records: list[pd.DataFrame] = []
        place_counter = 0

        for _, cell_df in groups:
            place_counter = self._process_cell(cell_df, records, place_counter)
            if self.pbar is not None:
                self.pbar.update(1)

        if not records:
            raise ValueError(
                "No places survived filtering — consider relaxing "
                "heading_tolerance or min_images_per_place."
            )

        result = pd.concat(records, ignore_index=True)
        result["place_id"] = pd.factorize(result["place_id"], sort=True)[0]
        return {
            **context,
            "traindataset": result.sort_values("place_id").reset_index(drop=True),
        }

    def _process_cell(
        self,
        cell_df: pd.DataFrame,
        records: list[pd.DataFrame],
        place_counter: int,
    ) -> int:
        """Classify images in one cell into lateral / frontal places.

        Returns the updated ``place_counter``.
        """
        if len(cell_df) < self.min_images_per_place:
            return place_counter

        coords = cell_df[["utm_east", "utm_north"]].values  # (p, 2)
        centroid = coords.mean(axis=0)
        centred = coords - centroid

        if centred.shape[0] < 2:
            return place_counter

        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        pc1 = Vt[0]  # ≈ road direction
        pc2 = Vt[1]  # ≈ roadside direction

        focal_points = [
            (centroid + self.focal_distance * pc2, "lat"),
            (centroid + self.focal_distance * pc1, "front"),
        ]

        for focal_point, _label in focal_points:
            delta_e = focal_point[0] - coords[:, 0]
            delta_n = focal_point[1] - coords[:, 1]
            alpha = np.degrees(np.arctan2(delta_e, delta_n)) % 360

            headings = cell_df["heading"].values % 360
            angular_diff = np.abs(alpha - headings)
            angular_diff = np.minimum(angular_diff, 360 - angular_diff)

            mask = angular_diff <= self.heading_tolerance
            if mask.sum() < self.min_images_per_place:
                continue

            sub = cell_df.iloc[np.where(mask)[0]].copy()
            sub["place_id"] = place_counter
            records.append(sub)
            place_counter += 1

        return place_counter


# ---------------------------------------------------------------------------
# Step 5 – coherence + diversity filtering on existing place IDs
# ---------------------------------------------------------------------------


class AssignDiversePlaceIdWithEmbedStep(BaseStep):
    """Remove incoherent then redundant images from each place.

    Runs three sequential phases on every place:

    **Phase 1 – Coherence**
        Iteratively drop the image with the lowest mean cosine similarity to
        its neighbours until every survivor is above ``cos_sim_threshold``.

    **Phase 2 – Diversity**
        Iteratively drop the more redundant member of the most-similar pair
        until no pair exceeds ``max_pair_similarity``.

    **Phase 3 – Cap** *(optional)*
        If the place still exceeds ``max_images_per_place``, keep removing
        the more redundant image of the closest pair.

    Places left with fewer than ``min_images`` survivors are removed entirely.

    Parameters
    ----------
    image_embedding_name:
        Sub-directory name under ``$PLACEFORGE_FEATURE_STORE_DIR/embedding/image/``.
    cos_sim_threshold:
        Minimum mean cosine similarity for an image to be retained (Phase 1).
    max_pair_similarity:
        Maximum pairwise cosine similarity allowed between any two images (Phase 2).
    max_images_per_place:
        Hard cap on images per place.  ``None`` disables the cap (Phase 3).
    min_images:
        Places with fewer surviving images than this are discarded entirely.
    """

    def __init__(
        self,
        image_embedding_name: str,
        cos_sim_threshold: float = 0.3,
        max_pair_similarity: float = 0.9,
        max_images_per_place: int | None = None,
        min_images: int = 2,
    ) -> None:
        super().__init__()
        self.cos_sim_threshold = cos_sim_threshold
        self.max_pair_similarity = max_pair_similarity
        self.max_images_per_place = max_images_per_place
        self.min_images = min_images
        self.image_cache = _image_cache(image_embedding_name)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()
        assert (
            "place_id" in df.columns
        ), "DiversifyPlacesStep expects place_id to already be assigned."

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

        # Phase 1 – coherence: drop the most dissimilar image iteratively
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

        # Phase 2 – diversity: drop the more redundant of the closest pair
        while len(alive) > 1:
            subset = normed[alive]
            sim = subset @ subset.T
            sim.fill_diagonal_(-1.0)

            if sim.max().item() <= self.max_pair_similarity:
                break

            i, j = divmod(int(sim.argmax()), len(alive))
            mean_i = sim[i].sum() / (len(alive) - 1)
            mean_j = sim[j].sum() / (len(alive) - 1)
            victim = i if mean_i >= mean_j else j

            dropped.add(alive[victim])
            alive.pop(victim)

        # Phase 3 – hard cap (optional)
        if self.max_images_per_place is not None:
            while len(alive) > self.max_images_per_place:
                subset = normed[alive]
                sim = subset @ subset.T
                sim.fill_diagonal_(-1.0)

                i, j = divmod(int(sim.argmax()), len(alive))
                mean_i = sim[i].sum() / (len(alive) - 1)
                mean_j = sim[j].sum() / (len(alive) - 1)
                victim = i if mean_i >= mean_j else j

                dropped.add(alive[victim])
                alive.pop(victim)

        return dropped
