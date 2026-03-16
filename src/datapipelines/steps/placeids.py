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


class AssignCosPlacePlaceIdStep(BaseStep):
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

        df["cell_x"] = (df["utm_east"] / self.cell_size_meters).apply(floor)
        df["cell_y"] = (df["utm_north"] / self.cell_size_meters).apply(floor)
        df["cell_h"] = (df["heading"] / self.heading_size_degrees).apply(floor)

        n_y = df["cell_y"].max() + 1
        n_h = df["cell_h"].max() + 1
        df["place_id"] = df["cell_x"] * (n_y * n_h) + df["cell_y"] * n_h + df["cell_h"]

        return {**context, "traindataset": df.sort_values("place_id").reset_index(drop=True)}


class AssignEigenPlacesPlaceIdStep(BaseStep):
    """Assign place IDs following the EigenPlaces training paradigm.
 
    For each spatial cell (defined by ``cell_size_meters``):
 
    1. Compute the SVD / PCA of the centred UTM coordinates of images
       in the cell.  The first principal component approximates the road
       direction; the second is perpendicular (towards the roadside).
 
    2. Define a **lateral focal point** at distance ``focal_distance``
       along the second principal component from the cell centroid, and
       a **frontal focal point** at the same distance along the first.
 
    3. For every image, compute the angle from its position to each
       focal point.  If the image's heading is within
       ``heading_tolerance`` degrees of that angle, assign it to the
       corresponding *lateral* or *frontal* place class for that cell.
 
    Images that face neither focal point are dropped.  Each (cell,
    lateral/frontal) pair becomes a distinct place ID.
 
    Parameters
    ----------
    cell_size_meters : float
        Side length of the square spatial cells (paper uses M = 15 m).
    focal_distance : float
        Distance *D* from the cell centroid to the focal point along
        the relevant principal component (paper default: 10 m).
    heading_tolerance : float
        Maximum angular deviation (degrees) between an image's heading
        and the computed angle to the focal point for the image to be
        included in that class.
    min_images_per_place : int
        Cells (per focal-point side) with fewer surviving images are
        discarded.
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
 
        # --- 1. Assign each image to a grid cell ---------------------------
        df["cell_x"] = (df["utm_east"] / self.cell_size_meters).apply(floor)
        df["cell_y"] = (df["utm_north"] / self.cell_size_meters).apply(floor)
        records: list[pd.DataFrame] = []
        place_counter = 0
 
        groups = list(df.groupby(["cell_x", "cell_y"]))
 
        if self.pbar is not None:
            self.pbar.reset(total=len(groups))
 
        for _, cell_df in groups:
            if len(cell_df) < self.min_images_per_place:
                if self.pbar is not None:
                    self.pbar.update(1)
                continue
 
            coords = cell_df[["utm_east", "utm_north"]].values  # (p, 2)
            centroid = coords.mean(axis=0)  # (2,)
            centred = coords - centroid  # (p, 2)
 
            # --- 2. SVD to get principal components -------------------------
            # centred = U @ diag(S) @ Vt   with Vt rows = principal directions
            if centred.shape[0] < 2:
                if self.pbar is not None:
                    self.pbar.update(1)
                continue
 
            _, _, Vt = np.linalg.svd(centred, full_matrices=False)
            pc1 = Vt[0]  # direction of maximum variance  (≈ road)
            pc2 = Vt[1]  # perpendicular direction         (≈ roadside)
 
            # --- 3. Focal points --------------------------------------------
            lateral_focal = centroid + self.focal_distance * pc2   # Eq. (1)
            frontal_focal = centroid + self.focal_distance * pc1
 
            for focal_point, side_label in [
                (lateral_focal, "lat"),
                (frontal_focal, "front"),
            ]:
                # Angle from each image to the focal point (Eq. 2)
                delta_e = focal_point[0] - coords[:, 0]
                delta_n = focal_point[1] - coords[:, 1]
                alpha = np.degrees(np.arctan2(delta_e, delta_n)) % 360  # 0-360
 
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
 
            if self.pbar is not None:
                self.pbar.update(1)
 
        if not records:
            raise ValueError(
                "No places survived filtering — consider relaxing "
                "heading_tolerance or min_images_per_place."
            )
 
        result = pd.concat(records, ignore_index=True)
        # Re-factorise place_id to be contiguous 0..K-1
        result["place_id"] = pd.factorize(result["place_id"], sort=True)[0]
        result = result.sort_values("place_id").reset_index(drop=True)
 
        return {**context, "traindataset": result}
 