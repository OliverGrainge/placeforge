from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision.io
from torch.utils.data import Dataset, Sampler


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_train_df(name: str) -> tuple[pd.DataFrame, Path]:
    """Load the train parquet and return (df, raw_dir)."""
    processed_dir = Path(os.environ["PLACEFORGE_PROCESSED_DIR"])
    raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
    parquet_path = processed_dir / "train" / name / "traindataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {parquet_path}")
    df = pd.read_parquet(parquet_path).set_index("image_id")
    return df, raw_dir


def subsample_geographic(
    df: pd.DataFrame,
    fraction: float = 1.0,
    tile_size: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Subsample a DataFrame by retaining a fraction of coarse spatial tiles.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``utm_east`` and ``utm_north`` columns.
    fraction : float
        Fraction of tiles to keep (0, 1].  1.0 returns the input unchanged.
    tile_size : float
        Side length (metres) of the coarse grid tiles.
    seed : int
        Random seed for reproducible tile selection.
    """
    if fraction >= 1.0:
        return df
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    tile_x = np.floor(df["utm_east"].values / tile_size).astype(np.int64)
    tile_y = np.floor(df["utm_north"].values / tile_size).astype(np.int64)
    tile_keys = set(zip(tile_x, tile_y))

    rng = np.random.RandomState(seed)
    tile_list = sorted(tile_keys)
    n_keep = max(1, int(len(tile_list) * fraction))
    keep_indices = rng.choice(len(tile_list), size=n_keep, replace=False)
    keep_tiles = {tile_list[i] for i in keep_indices}

    mask = np.array([(tx, ty) in keep_tiles for tx, ty in zip(tile_x, tile_y)])
    return df[mask].copy()


# ---------------------------------------------------------------------------
# Contrastive training (place-level sampling)
# ---------------------------------------------------------------------------


class ContrastiveBatchSampler(Sampler):
    """Yields batches where all samples share the same supergroup.

    Batches from all supergroups are collected and then globally shuffled,
    so supergroups freely interleave at the batch level.  Within each
    supergroup, places are shuffled without replacement.
    """

    def __init__(
        self,
        supergroup_to_indices: dict[Any, list[int]],
        batch_size: int,
        drop_last: bool = False,
    ):
        self.supergroup_to_indices = supergroup_to_indices
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        all_batches = []
        for indices in self.supergroup_to_indices.values():
            sampled = indices.copy()
            random.shuffle(sampled)
            for i in range(0, len(sampled), self.batch_size):
                batch = sampled[i : i + self.batch_size]
                if not self.drop_last or len(batch) == self.batch_size:
                    all_batches.append(batch)
        random.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        if self.drop_last:
            return sum(
                len(idx) // self.batch_size
                for idx in self.supergroup_to_indices.values()
            )
        return sum(
            math.ceil(len(idx) / self.batch_size)
            for idx in self.supergroup_to_indices.values()
        )


class ContrastiveTrainDataset(Dataset):
    """Place-level train dataset for contrastive learning.

    Each sample is a *place*: ``images_per_place`` images are randomly drawn
    from that place.  The collate function flattens them into a single batch
    so the loss can identify positives by shared ``place_id``.

    Note: ``place_id`` in the saved parquet is already 0-indexed within each
    supergroup (remapped by ``SaveTrainDataset``), so it is used directly as
    the local label.
    """

    def __init__(
        self,
        name: str,
        images_per_place: int,
        transform: Any = None,
        fraction: float = 1.0,
    ):
        self.images_per_place = images_per_place
        self.transform = transform
        self.df, self.raw_dir = _load_train_df(name)
        self.df = subsample_geographic(self.df, fraction=fraction)

        self._has_heading = "heading" in self.df.columns
        self._has_utm = "utm_east" in self.df.columns and "utm_north" in self.df.columns

        self._group_cols = ["supergroup_id", "place_id"]
        self.place_ids = (
            self.df[self._group_cols]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        self.place_ids = list(self.place_ids)  # list of (supergroup_id, place_id)
        self.place_id_to_paths = (
            self.df.groupby(self._group_cols)["image_path"].apply(list).to_dict()
        )
        if self._has_heading:
            self.place_id_to_headings = (
                self.df.groupby(self._group_cols)["heading"].apply(list).to_dict()
            )
        if self._has_utm:
            self.place_id_to_utm_east = (
                self.df.groupby(self._group_cols)["utm_east"].apply(list).to_dict()
            )
            self.place_id_to_utm_north = (
                self.df.groupby(self._group_cols)["utm_north"].apply(list).to_dict()
            )
        self._supergroup_to_indices = self._build_supergroup_to_indices()

    def __len__(self) -> int:
        return len(self.place_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        supergroup_id, place_id = self.place_ids[idx]
        key = (supergroup_id, place_id)
        all_paths = self.place_id_to_paths[key]
        n = self.images_per_place
        if n <= len(all_paths):
            indices = random.sample(range(len(all_paths)), k=n)
        else:
            indices = random.choices(range(len(all_paths)), k=n)

        sampled_paths = [all_paths[i] for i in indices]

        images = []
        for path in sampled_paths:
            image = torchvision.io.read_image(
                str(self.raw_dir / path), mode=torchvision.io.ImageReadMode.RGB
            )
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        result = {
            "images": images,
            "place_id": place_id,
            "local_label": place_id,
            "supergroup_id": supergroup_id,
        }
        if self._has_heading:
            all_headings = self.place_id_to_headings[key]
            result["headings"] = [all_headings[i] for i in indices]
        if self._has_utm:
            all_utm_east = self.place_id_to_utm_east[key]
            all_utm_north = self.place_id_to_utm_north[key]
            result["utm"] = [[all_utm_east[i], all_utm_north[i]] for i in indices]
        return result

    @property
    def num_supergroups(self) -> int:
        return self.df["supergroup_id"].nunique()

    @property
    def num_places(self) -> int:
        return len(self.place_ids)

    @property
    def num_images(self) -> int:
        return len(self.df)

    def _build_supergroup_to_indices(self) -> dict[Any, list[int]]:
        sg_to_idx: dict[Any, list[int]] = defaultdict(list)
        for idx, (supergroup_id, _place_id) in enumerate(self.place_ids):
            sg_to_idx[supergroup_id].append(idx)
        return dict(sg_to_idx)

    @property
    def supergroup_to_indices(self) -> dict[Any, list[int]]:
        return self._supergroup_to_indices

    @property
    def supergroup_num_places(self) -> dict[Any, int]:
        return {sg: len(indices) for sg, indices in self._supergroup_to_indices.items()}

    def get_batch_sampler(
        self, batch_size: int, drop_last: bool = False,
    ) -> ContrastiveBatchSampler:
        """Return a batch sampler with globally-shuffled supergroup batches."""
        return ContrastiveBatchSampler(
            self.supergroup_to_indices,
            batch_size,
            drop_last=drop_last,
        )

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate individual place samples into a single batch dict.

        Returns:
            images: tensor of shape (B * images_per_place, C, H, W)
            place_ids: tensor of shape (B * images_per_place,), one per image
            supergroup_id: scalar tensor, shared by all images in the batch
            headings: tensor of shape (B * images_per_place,), if available
        """
        supergroup_ids = [sample["supergroup_id"] for sample in batch]
        unique_supergroup_ids = set(supergroup_ids)
        if len(unique_supergroup_ids) != 1:
            raise AssertionError(
                "Train batch mixed supergroup_ids: "
                f"{sorted(unique_supergroup_ids)}"
            )

        images = torch.stack([img for sample in batch for img in sample["images"]])
        place_ids = [
            pid
            for sample in batch
            for pid in [sample["local_label"]] * len(sample["images"])
        ]
        result = {
            "images": images,
            "place_ids": torch.tensor(place_ids),
            "supergroup_id": torch.tensor(supergroup_ids[0]),
        }

        return result
