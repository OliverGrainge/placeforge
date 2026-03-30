from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from .base import BaseStep


def _make_sector(
    center: np.ndarray,
    heading: float,
    radius: float,
    half_fov: float,
    n_arc_points: int = 64,
) -> Polygon:
    """Build a polygon approximating a circular sector.

    Uses compass convention: east = r·sin(θ), north = r·cos(θ).
    """
    angles = np.linspace(heading - half_fov, heading + half_fov, n_arc_points)
    arc_e = center[0] + radius * np.sin(angles)
    arc_n = center[1] + radius * np.cos(angles)
    coords = np.empty((n_arc_points + 2, 2))
    coords[0] = center
    coords[1:-1, 0] = arc_e
    coords[1:-1, 1] = arc_n
    coords[-1] = center
    return Polygon(coords)


def _build_iou_lookup(
    radius: float,
    half_fov: float,
    grid_resolution: int,
) -> RegularGridInterpolator:
    """Precompute a 3D lookup table of IoU(distance, rel_angle_a, rel_angle_b).

    By rotational symmetry the IoU of two identical-shape sectors depends only
    on the distance *d* between centres and each sector's heading relative to
    the bearing between them.  We place the anchor at the origin and the
    neighbour at (0, d) (due north), sweep all heading combinations on a grid,
    and store the results for fast interpolation at runtime.
    """
    n = grid_resolution
    d_vals = np.linspace(0, 2 * radius, n)
    angle_vals = np.linspace(-np.pi, np.pi, n)

    grid = np.zeros((n, n, n))
    origin = np.array([0.0, 0.0])

    for i_d, d in enumerate(d_vals):
        neighbor_pos = np.array([0.0, d])
        for i_a, rel_a in enumerate(angle_vals):
            anchor = _make_sector(origin, rel_a, radius, half_fov)
            anchor_area = shapely.area(anchor)

            neighbors = np.array([
                _make_sector(neighbor_pos, rel_b, radius, half_fov)
                for rel_b in angle_vals
            ])

            inter_areas = shapely.area(shapely.intersection(anchor, neighbors))
            grid[i_d, i_a, :] = np.where(
                inter_areas > 0,
                inter_areas / (2.0 * anchor_area - inter_areas),
                0.0,
            )

    return RegularGridInterpolator(
        (d_vals, angle_vals, angle_vals),
        grid,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )


class ComputeFoVOverlapStep(BaseStep):
    """Compute pairwise FoV overlap (IoU) for nearby images.

    Uses UTM positions and compass headings to model each camera's field of
    view as a circular sector.  A 3D lookup table of IoU values is
    precomputed once (keyed by distance and relative headings), then every
    pair is resolved via fast trilinear interpolation — no per-pair polygon
    intersection at runtime.

    Builds the KDTree **once** and processes images sequentially in small
    batches to keep memory usage constant regardless of dataset size.

    Adds ``context["graded_pairs"]``: a DataFrame with columns
    ``anchor_idx``, ``pair_idx``, ``similarity`` (only pairs with IoU > 0).
    """

    def __init__(
        self,
        fov_radius: float = 25.0,
        fov_angle: float = 90.0,
        grid_resolution: int = 64,
        batch_size: int = 1024,
    ) -> None:
        super().__init__()
        self.fov_radius = fov_radius
        self.fov_angle = fov_angle
        self.grid_resolution = grid_resolution
        self.batch_size = batch_size
        self._iou_interp = _build_iou_lookup(
            radius=fov_radius,
            half_fov=np.deg2rad(fov_angle) / 2,
            grid_resolution=grid_resolution,
        )

    def _compute_fov_iou_batch(
        self,
        anchor_pos: np.ndarray,
        anchor_heading: float,
        neighbor_pos: np.ndarray,
        neighbor_headings: np.ndarray,
    ) -> np.ndarray:
        """Look up IoU for one anchor vs M neighbors via interpolation."""
        dx = neighbor_pos[:, 0] - anchor_pos[0]
        dy = neighbor_pos[:, 1] - anchor_pos[1]
        d = np.sqrt(dx**2 + dy**2)
        bearing = np.arctan2(dx, dy)  # compass bearing

        rel_a = (anchor_heading - bearing + np.pi) % (2 * np.pi) - np.pi
        rel_b = (neighbor_headings - bearing + np.pi) % (2 * np.pi) - np.pi

        points = np.column_stack([d, rel_a, rel_b])
        return np.maximum(self._iou_interp(points), 0.0)

    def cache_params(self) -> dict[str, Any]:
        return {
            "fov_radius": self.fov_radius,
            "fov_angle": self.fov_angle,
            "grid_resolution": self.grid_resolution,
        }

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"]

        for col in ("utm_east", "utm_north", "heading"):
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in traindataset")

        valid_mask = df["utm_east"].notna() & df["utm_north"].notna() & df["heading"].notna()
        if not valid_mask.all():
            df = df[valid_mask].reset_index(drop=True)

        coords = df[["utm_east", "utm_north"]].values.astype(np.float64)
        headings_rad = np.deg2rad(df["heading"].values.astype(np.float64))
        search_radius = 2.0 * self.fov_radius

        n_images = len(df)

        # Build the KDTree once — O(N log N) time, single copy of coords
        tree = cKDTree(coords)

        all_anchors: list[np.ndarray] = []
        all_pairs: list[np.ndarray] = []
        all_sims: list[np.ndarray] = []

        if self.pbar is not None:
            self.pbar.reset(total=n_images)

        # Process in small batches to keep memory flat
        for batch_start in range(0, n_images, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_images)
            batch_coords = coords[batch_start:batch_end]

            # Batch neighbor query — returns list[list[int]]
            neighbors_list = tree.query_ball_point(batch_coords, r=search_radius)

            for local_i, neighbors in enumerate(neighbors_list):
                global_i = batch_start + local_i
                # Only keep j > global_i to avoid duplicate pairs
                valid_j = np.array(
                    [j for j in neighbors if j > global_i], dtype=np.int64
                )
                if len(valid_j) == 0:
                    continue

                ious = self._compute_fov_iou_batch(
                    anchor_pos=coords[global_i],
                    anchor_heading=headings_rad[global_i],
                    neighbor_pos=coords[valid_j],
                    neighbor_headings=headings_rad[valid_j],
                )

                nonzero = ious > 0
                if nonzero.any():
                    n_nz = int(nonzero.sum())
                    all_anchors.append(
                        np.full(n_nz, global_i, dtype=np.int64)
                    )
                    all_pairs.append(valid_j[nonzero])
                    all_sims.append(ious[nonzero])

            if self.pbar is not None:
                self.pbar.update(batch_end - batch_start)

        if all_anchors:
            pairs_df = pd.DataFrame({
                "anchor_idx": np.concatenate(all_anchors),
                "pair_idx": np.concatenate(all_pairs),
                "similarity": np.concatenate(all_sims),
            })
        else:
            pairs_df = pd.DataFrame(columns=["anchor_idx", "pair_idx", "similarity"])

        context["traindataset"] = df
        context["graded_pairs"] = pairs_df
        return context


class SaveGradedTrainDataset(BaseStep):
    """Save the image-level parquet and the precomputed FoV-overlap pairs."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        processed_dir = os.environ["PLACEFORGE_PROCESSED_DIR"]
        output_dir = Path(processed_dir) / "train" / self.name

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        df = context["traindataset"]
        df.to_parquet(output_dir / "traindataset.parquet")

        pairs_df = context["graded_pairs"]
        pairs_df.to_parquet(output_dir / "pairs.parquet", index=False)

        return context
