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


def _load_train_df(
    name: str,
    num_supergroups: int | None = None,
    num_places: int | None = None,
    min_places_per_supergroup: int | None = None,
    order_by: str | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load the train parquet and return (df, raw_dir).

    At most one of *num_supergroups* and *num_places* may be given.
    *num_supergroups* keeps the first N supergroups (by id).
    *num_places* keeps a deterministic random subset of N places (seeded by
    *num_places* so the same count always produces the same subset).

    *min_places_per_supergroup* guarantees each supergroup contributes at least
    this many places (capped by how many the supergroup actually has).  When
    combined with *num_places*, the minimum is satisfied first and the remaining
    budget is distributed across supergroups.

    *order_by* is a column name in the parquet.  When given, places within each
    supergroup are ranked by the minimum value of that column (ascending) and
    selected in that order instead of randomly.
    """
    if num_supergroups is not None and num_places is not None:
        raise ValueError("Only one of num_supergroups and num_places may be specified")
    processed_dir = Path(os.environ["PLACEFORGE_PROCESSED_DIR"])
    raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
    parquet_path = processed_dir / "train" / name / "traindataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {parquet_path}")
    df = pd.read_parquet(parquet_path).set_index("image_id")
    if num_supergroups is not None:
        keep = sorted(df["supergroup_id"].unique())[:num_supergroups]
        df = df[df["supergroup_id"].isin(keep)]
    if num_places is not None:
        place_df = df[["supergroup_id", "place_id"]].drop_duplicates()

        # Build ordered place lists per supergroup
        if order_by is not None:
            if order_by not in df.columns:
                raise ValueError(
                    f"order_by column '{order_by}' not found in dataset. "
                    f"Available columns: {sorted(df.columns.tolist())}"
                )
            # Rank places by the minimum value of the order_by column
            place_rank = (
                df.groupby(["supergroup_id", "place_id"])[order_by]
                .min()
                .reset_index()
                .sort_values(["supergroup_id", order_by])
            )
            grouped = {
                sg: g["place_id"].tolist()
                for sg, g in place_rank.groupby("supergroup_id")
            }
        else:
            grouped = {
                sg: sorted(g["place_id"].tolist())
                for sg, g in place_df.groupby("supergroup_id")
            }

        n_supergroups = len(grouped)
        min_per_sg = min_places_per_supergroup or 0

        # Phase 1: allocate the minimum to each supergroup
        base_alloc = {
            sg: min(min_per_sg, len(places))
            for sg, places in grouped.items()
        }
        allocated = sum(base_alloc.values())
        remaining = max(0, num_places - allocated)

        # Phase 2: distribute the remaining budget equally
        if remaining > 0:
            headroom = {
                sg: len(grouped[sg]) - base_alloc[sg]
                for sg in sorted(grouped)
            }
            per_sg_extra = remaining // n_supergroups
            extra_remainder = remaining % n_supergroups
            for i, sg in enumerate(sorted(grouped)):
                extra = min(per_sg_extra + (1 if i < extra_remainder else 0), headroom[sg])
                base_alloc[sg] += extra

        # Phase 3: select places (ordered or random)
        rng = random.Random(num_places)
        sampled: list[tuple] = []
        for sg in sorted(grouped):
            places = grouped[sg]
            k = base_alloc[sg]
            if order_by is not None:
                # Places are already sorted by the order_by column
                selected = places[:k]
            else:
                selected = rng.sample(places, k)
            sampled.extend((sg, pid) for pid in selected)

        keep_set = set(sampled)
        mask = np.array([
            (sg, pid) in keep_set
            for sg, pid in zip(df["supergroup_id"].values, df["place_id"].values)
        ])
        df = df[mask]
    elif min_places_per_supergroup is not None:
        # Apply min_places_per_supergroup as a filter even without num_places
        place_df = df[["supergroup_id", "place_id"]].drop_duplicates()
        sg_counts = place_df.groupby("supergroup_id").size()
        keep_sgs = sg_counts[sg_counts >= min_places_per_supergroup].index
        df = df[df["supergroup_id"].isin(keep_sgs)]
    return df, raw_dir


# ---------------------------------------------------------------------------
# Contrastive training (place-level sampling)
# ---------------------------------------------------------------------------


class ContrastiveBatchSampler(Sampler):
    """Yields batches where all samples share the same supergroup.

    Batches from all supergroups are collected and then globally shuffled,
    so supergroups freely interleave at the batch level.
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
            shuffled = indices.copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i : i + self.batch_size]
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
        num_supergroups: int | None = None,
        num_places: int | None = None,
        min_places_per_supergroup: int | None = None,
        order_by: str | None = None,
    ):
        self.images_per_place = images_per_place
        self.transform = transform
        self.df, self.raw_dir = _load_train_df(
            name, num_supergroups, num_places,
            min_places_per_supergroup=min_places_per_supergroup,
            order_by=order_by,
        )

        self._has_heading = "heading" in self.df.columns
        self._has_utm = "utm_east" in self.df.columns and "utm_north" in self.df.columns

        self.place_ids = self.df["place_id"].unique()
        self.place_id_to_paths = (
            self.df.groupby("place_id")["image_path"].apply(list).to_dict()
        )
        if self._has_heading:
            self.place_id_to_headings = (
                self.df.groupby("place_id")["heading"].apply(list).to_dict()
            )
        if self._has_utm:
            self.place_id_to_utm_east = (
                self.df.groupby("place_id")["utm_east"].apply(list).to_dict()
            )
            self.place_id_to_utm_north = (
                self.df.groupby("place_id")["utm_north"].apply(list).to_dict()
            )
        self.place_id_to_supergroup = (
            self.df.groupby("place_id")["supergroup_id"].first().to_dict()
        )
        self._supergroup_to_indices = self._build_supergroup_to_indices()

    def __len__(self) -> int:
        return len(self.place_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        place_id = self.place_ids[idx]
        all_paths = self.place_id_to_paths[place_id]
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
            "supergroup_id": self.place_id_to_supergroup[place_id],
        }
        if self._has_heading:
            all_headings = self.place_id_to_headings[place_id]
            result["headings"] = [all_headings[i] for i in indices]
        if self._has_utm:
            all_utm_east = self.place_id_to_utm_east[place_id]
            all_utm_north = self.place_id_to_utm_north[place_id]
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
        for idx, place_id in enumerate(self.place_ids):
            sg_to_idx[self.place_id_to_supergroup[place_id]].append(idx)
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
            self.supergroup_to_indices, batch_size, drop_last=drop_last,
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
        if "headings" in batch[0]:
            result["headings"] = torch.tensor(
                [h for sample in batch for h in sample["headings"]]
            )
        if "utm" in batch[0]:
            result["utm"] = torch.tensor(
                [coord for sample in batch for coord in sample["utm"]]
            )
        return result


# ---------------------------------------------------------------------------
# Classification training (image-level sampling)
# ---------------------------------------------------------------------------


class ClassificationTrainDataset(Dataset):
    """Image-level train dataset for classification.

    Every image in the dataset is visited once per epoch with standard random
    shuffling (no supergroup ordering).  Each image carries a globally unique
    0-indexed label suitable for a single CosFace head.
    """

    def __init__(
        self,
        name: str,
        transform: Any = None,
        num_supergroups: int | None = None,
        num_places: int | None = None,
        min_places_per_supergroup: int | None = None,
        order_by: str | None = None,
    ):
        self.transform = transform
        self.df, self.raw_dir = _load_train_df(
            name, num_supergroups, num_places,
            min_places_per_supergroup=min_places_per_supergroup,
            order_by=order_by,
        )

        self._has_heading = "heading" in self.df.columns
        self._has_utm = "utm_east" in self.df.columns and "utm_north" in self.df.columns

        # Pre-compute flat arrays for fast __getitem__
        self._image_paths: list[str] = self.df["image_path"].tolist()
        if self._has_heading:
            self._headings = self.df["heading"].values
        if self._has_utm:
            self._utm = np.stack([self.df["utm_east"].values, self.df["utm_north"].values], axis=1)

        # Build globally unique 0-indexed labels from (supergroup_id, place_id)
        sg_ids = self.df["supergroup_id"].values
        place_ids = self.df["place_id"].values
        unique_pairs = sorted(set(zip(sg_ids, place_ids)))
        pair_to_label = {pair: i for i, pair in enumerate(unique_pairs)}
        self._labels = np.array(
            [pair_to_label[(sg, pid)] for sg, pid in zip(sg_ids, place_ids)],
            dtype=np.int64,
        )
        self._num_places = len(unique_pairs)

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image = torchvision.io.read_image(
            str(self.raw_dir / self._image_paths[idx]),
            mode=torchvision.io.ImageReadMode.RGB,
        )
        if self.transform is not None:
            image = self.transform(image)

        result = {
            "image": image,
            "label": int(self._labels[idx]),
        }
        if self._has_heading:
            result["heading"] = float(self._headings[idx])
        if self._has_utm:
            result["utm"] = self._utm[idx].tolist()
        return result

    @property
    def num_places(self) -> int:
        return self._num_places

    @property
    def num_images(self) -> int:
        return len(self._image_paths)

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        result = {
            "images": torch.stack([s["image"] for s in batch]),
            "labels": torch.tensor([s["label"] for s in batch]),
        }
        if "heading" in batch[0]:
            result["headings"] = torch.tensor([s["heading"] for s in batch])
        if "utm" in batch[0]:
            result["utm"] = torch.tensor([s["utm"] for s in batch])
        return result
