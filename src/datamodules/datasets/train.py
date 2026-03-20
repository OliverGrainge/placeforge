from __future__ import annotations

import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import pandas as pd
import torchvision.io
from torch.utils.data import Dataset, Sampler


class TrainDataset(Dataset):
    def __init__(self, name: str, images_per_place: int, transform: Any = None):
        self.images_per_place = images_per_place
        self.transform = transform
        self.df = self._load_df(name)

        self.place_ids = self.df["place_id"].unique()
        self.place_id_to_paths = (
            self.df.groupby("place_id")["image_path"].apply(list).to_dict()
        )
        self.place_id_to_supergroup = (
            self.df.groupby("place_id")["supergroup_id"].first().to_dict()
        )
        self._supergroup_to_indices = self._build_supergroup_to_indices()

    def _load_df(self, name: str) -> pd.DataFrame:
        processed_dir = Path(os.environ["PLACEFORGE_PROCESSED_DIR"])
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
        self.dataset_dir = processed_dir / "train" / name
        self.parquet_path = self.dataset_dir / "traindataset.parquet"

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Train dataset not found: {self.parquet_path}")

        return pd.read_parquet(self.parquet_path).set_index("image_id")

    def __len__(self) -> int:
        return len(self.place_ids)

    def __getitem__(self, image_id: int) -> dict[str, Any]:
        place_id = self.place_ids[image_id]
        all_paths = self.place_id_to_paths[place_id]
        sampled_paths = random.choices(all_paths, k=self.images_per_place)

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
        self, batch_size: int, drop_last: bool = False, sequential: bool = False
    ) -> SupergroupBatchSampler:
        """Return a batch sampler that groups samples by supergroup."""
        return SupergroupBatchSampler(
            self.supergroup_to_indices, batch_size, drop_last=drop_last, sequential=sequential
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
            for pid in [sample["place_id"]] * len(sample["images"])
        ]
        return {
            "images": images,
            "place_ids": torch.tensor(place_ids),
            "supergroup_id": torch.tensor(supergroup_ids[0]),
        }


class SupergroupBatchSampler(Sampler):
    """Yields batches of indices where all samples are from the same supergroup.

    When ``sequential=False`` (default) all batches are globally shuffled,
    freely interleaving supergroups.  When ``sequential=True`` the supergroup
    order is shuffled once, then each supergroup is fully exhausted before
    moving to the next.
    """

    def __init__(
        self,
        supergroup_to_image_id: dict[Any, list[int]],
        batch_size: int,
        drop_last: bool = False,
        sequential: bool = False,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sequential = sequential
        self.supergroup_to_indices = supergroup_to_image_id

    def __iter__(self):
        supergroups = list(self.supergroup_to_indices.keys())
        random.shuffle(supergroups)

        if self.sequential:
            for sg in supergroups:
                shuffled = self.supergroup_to_indices[sg].copy()
                random.shuffle(shuffled)
                for i in range(0, len(shuffled), self.batch_size):
                    batch = shuffled[i : i + self.batch_size]
                    if not self.drop_last or len(batch) == self.batch_size:
                        yield batch
        else:
            all_batches = []
            for sg in supergroups:
                shuffled = self.supergroup_to_indices[sg].copy()
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
                len(indices) // self.batch_size
                for indices in self.supergroup_to_indices.values()
            )
        return sum(
            math.ceil(len(indices) / self.batch_size)
            for indices in self.supergroup_to_indices.values()
        )
