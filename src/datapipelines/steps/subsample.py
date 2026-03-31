"""Geographic subsampling step.

Divides the map into coarse spatial tiles and randomly retains a fraction
of them, preserving local image density for all downstream steps.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseStep


class SubsampleGeographicStep(BaseStep):
    """Subsample images by retaining a fraction of coarse spatial tiles.

    Parameters
    ----------
    fraction : float
        Fraction of tiles to keep (0, 1].  ``1.0`` keeps everything.
    tile_size : float
        Side length (metres) of the coarse grid tiles used for grouping.
    seed : int
        Random seed for reproducible tile selection.
    """

    def __init__(
        self,
        fraction: float = 1.0,
        tile_size: float = 100.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self.fraction = fraction
        self.tile_size = tile_size
        self.seed = seed

    def cache_params(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "tile_size": self.tile_size,
            "seed": self.seed,
        }

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"]

        if self.fraction >= 1.0:
            return context

        # Assign each image to a coarse tile
        tile_x = np.floor(df["utm_east"].values / self.tile_size).astype(np.int64)
        tile_y = np.floor(df["utm_north"].values / self.tile_size).astype(np.int64)
        tile_keys = set(zip(tile_x, tile_y))

        # Randomly select a fraction of tiles
        rng = np.random.RandomState(self.seed)
        tile_list = sorted(tile_keys)
        n_keep = max(1, int(len(tile_list) * self.fraction))
        keep_indices = rng.choice(len(tile_list), size=n_keep, replace=False)
        keep_tiles = {tile_list[i] for i in keep_indices}

        # Filter to images in kept tiles
        mask = np.array([
            (tx, ty) in keep_tiles
            for tx, ty in zip(tile_x, tile_y)
        ])
        df = df[mask].copy()

        # Reset image_id to 0..N-1
        df["image_id"] = np.arange(len(df))

        n_before = len(mask)
        n_after = len(df)
        if self.pbar is not None:
            self.pbar.write(
                f"SubsampleGeographic: {n_after}/{n_before} images "
                f"({n_after / n_before * 100:.1f}%), "
                f"{n_keep}/{len(tile_list)} tiles kept"
            )

        return {**context, "traindataset": df}
