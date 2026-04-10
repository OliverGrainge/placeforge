from __future__ import annotations

import os
from typing import Any, List

import torch
from torch.utils.data import DataLoader

from .datamodule import BaseDataModule, _dataloader_worker_init_fn
from .datasets import ContrastiveTrainDataset, subsample_geographic


class CuraVPRTrainDataModule(BaseDataModule):
    def __init__(
        self,
        batch_size: int,
        train_dataset_name: str | None = None,
        val_dataset_names: List[str] | None = None,
        num_workers: int = os.cpu_count() // 2,
        images_per_place: int = 4,
        fraction: float = 1.0,
        train_transform: Any = None,
        val_transform: Any = None,
        test_dataset_names: List[str] | None = None,
        test_transform: Any = None,
        source_sample_weights: dict[str, float] | None = None,
    ):
        super().__init__(
            val_dataset_names=val_dataset_names,
            batch_size=batch_size,
            num_workers=num_workers,
            val_transform=val_transform,
            test_dataset_names=test_dataset_names,
            test_transform=test_transform,
        )
        self.train_dataset_name = train_dataset_name
        self.images_per_place = images_per_place
        self.fraction = fraction
        self.train_transform = train_transform
        self.source_sample_weights = source_sample_weights
        self.save_hyperparameters()

        self._train_dataset: ContrastiveTrainDataset | None = None

    @property
    def num_supergroups(self) -> int:
        if self._train_dataset is None:
            raise RuntimeError("setup() must be called before accessing num_supergroups")
        return self._train_dataset.num_supergroups

    @property
    def supergroup_num_places(self) -> dict:
        if self._train_dataset is None:
            raise RuntimeError("setup() must be called before accessing supergroup_num_places")
        return self._train_dataset.supergroup_num_places

    @property
    def num_places(self) -> int:
        if self._train_dataset is None:
            raise RuntimeError("setup() must be called before accessing num_places")
        return self._train_dataset.num_places

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            if not self.train_dataset_name:
                raise ValueError("train_dataset_name is required for fit setup")
            self._train_dataset = ContrastiveTrainDataset(
                self.train_dataset_name,
                images_per_place=self.images_per_place,
                transform=self.train_transform,
                fraction=self.fraction,
                source_sample_weights=self.source_sample_weights,
            )
        super().setup(stage)

    def train_dataloader(self) -> DataLoader:
        batch_sampler = self._train_dataset.get_batch_sampler(
            self.batch_size, drop_last=True,
        )
        return DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self._train_dataset.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=_dataloader_worker_init_fn if self.num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
