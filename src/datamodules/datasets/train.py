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
    name: str, num_supergroups: int | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Load the train parquet and return (df, raw_dir).

    If *num_supergroups* is given, only the first *num_supergroups*
    supergroups (by id) are kept.
    """
    processed_dir = Path(os.environ["PLACEFORGE_PROCESSED_DIR"])
    raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
    parquet_path = processed_dir / "train" / name / "traindataset.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {parquet_path}")
    df = pd.read_parquet(parquet_path).set_index("image_id")
    if num_supergroups is not None:
        keep = sorted(df["supergroup_id"].unique())[:num_supergroups]
        df = df[df["supergroup_id"].isin(keep)]
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
    ):
        self.images_per_place = images_per_place
        self.transform = transform
        self.df, self.raw_dir = _load_train_df(name, num_supergroups)

        self.place_ids = self.df["place_id"].unique()
        self.place_id_to_paths = (
            self.df.groupby("place_id")["image_path"].apply(list).to_dict()
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
            sampled_paths = random.sample(all_paths, k=n)
        else:
            sampled_paths = random.choices(all_paths, k=n)

        images = []
        for path in sampled_paths:
            image = torchvision.io.read_image(
                str(self.raw_dir / path), mode=torchvision.io.ImageReadMode.RGB
            )
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        return {
            "images": images,
            "place_id": place_id,
            "local_label": place_id,
            "supergroup_id": self.place_id_to_supergroup[place_id],
        }

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
        return {
            "images": images,
            "place_ids": torch.tensor(place_ids),
            "supergroup_id": torch.tensor(supergroup_ids[0]),
        }


# ---------------------------------------------------------------------------
# Classification training (image-level sampling)
# ---------------------------------------------------------------------------


class ClassificationBatchSampler(Sampler):
    """Yields batches where all samples share the same supergroup.

    Supergroup order is shuffled once, then each supergroup is fully exhausted
    before moving to the next.  This is required for classification training
    where each supergroup has its own classifier head.
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
        supergroups = list(self.supergroup_to_indices.keys())
        random.shuffle(supergroups)
        for sg in supergroups:
            shuffled = self.supergroup_to_indices[sg].copy()
            random.shuffle(shuffled)
            for i in range(0, len(shuffled), self.batch_size):
                batch = shuffled[i : i + self.batch_size]
                if not self.drop_last or len(batch) == self.batch_size:
                    yield batch

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


class ClassificationTrainDataset(Dataset):
    """Image-level train dataset for classification.

    Every image in the dataset is visited once per epoch.  Batches are
    constrained to a single supergroup via :class:`ClassificationBatchSampler`
    which exhausts all images in one supergroup before moving to the next.
    Each image carries a per-supergroup local label suitable for CosFace.

    Note: ``place_id`` in the saved parquet is already 0-indexed within each
    supergroup (remapped by ``SaveTrainDataset``), so it can be used directly
    as the local label without further remapping.
    """

    def __init__(self, name: str, transform: Any = None, num_supergroups: int | None = None):
        self.transform = transform
        self.df, self.raw_dir = _load_train_df(name, num_supergroups)

        # Pre-compute flat arrays for fast __getitem__
        self._image_paths: list[str] = self.df["image_path"].tolist()
        # place_id is already 0-indexed per supergroup (from SaveTrainDataset)
        self._local_labels = self.df["place_id"].values.astype(np.int64)
        self._supergroup_ids = self.df["supergroup_id"].values.astype(np.int64)

        # Supergroup -> image indices (row positions in the df)
        self._supergroup_to_indices: dict[Any, list[int]] = defaultdict(list)
        for idx, sg in enumerate(self._supergroup_ids):
            self._supergroup_to_indices[int(sg)].append(idx)
        self._supergroup_to_indices = dict(self._supergroup_to_indices)

        # Place counts per supergroup (for classifier sizing)
        self._supergroup_num_places: dict[Any, int] = {
            sg: len(set(self._local_labels[idxs]))
            for sg, idxs in self._supergroup_to_indices.items()
        }

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        image = torchvision.io.read_image(
            str(self.raw_dir / self._image_paths[idx]),
            mode=torchvision.io.ImageReadMode.RGB,
        )
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "local_label": int(self._local_labels[idx]),
            "supergroup_id": int(self._supergroup_ids[idx]),
        }

    @property
    def num_places(self) -> int:
        return sum(self._supergroup_num_places.values())

    @property
    def num_images(self) -> int:
        return len(self._image_paths)

    @property
    def num_supergroups(self) -> int:
        return len(self._supergroup_to_indices)

    @property
    def supergroup_num_places(self) -> dict[Any, int]:
        return self._supergroup_num_places

    def get_batch_sampler(
        self, batch_size: int, drop_last: bool = False,
    ) -> ClassificationBatchSampler:
        """Return a batch sampler with sequential supergroups."""
        return ClassificationBatchSampler(
            self._supergroup_to_indices, batch_size, drop_last=drop_last,
        )

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate single-image samples.

        Returns:
            images: (B, C, H, W)
            place_ids: (B,)
            supergroup_id: scalar tensor
        """
        supergroup_ids = [s["supergroup_id"] for s in batch]
        if len(set(supergroup_ids)) != 1:
            raise AssertionError(
                f"Train batch mixed supergroup_ids: {sorted(set(supergroup_ids))}"
            )
        return {
            "images": torch.stack([s["image"] for s in batch]),
            "place_ids": torch.tensor([s["local_label"] for s in batch]),
            "supergroup_id": torch.tensor(supergroup_ids[0]),
        }
