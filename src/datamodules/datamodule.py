from __future__ import annotations

import os
from typing import Any, List

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from . import register_datamodule
from .datasets import TrainDataset, ValDataset


@register_datamodule("place_recognition")
class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset_name: str,
        val_dataset_names: List[str],
        batch_size: int,
        num_workers: int = os.cpu_count() // 2,
        images_per_place: int = 4,
        train_transform: Any = None,
        val_transform: Any = None,
    ):
        super().__init__()
        self.train_dataset_name = train_dataset_name
        self.val_dataset_names = val_dataset_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.images_per_place = images_per_place
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.save_hyperparameters()

        self._train_dataset: TrainDataset | None = None
        self._val_datasets: List[ValDataset] = []

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._train_dataset = TrainDataset(
                self.train_dataset_name,
                images_per_place=self.images_per_place,
                transform=self.train_transform,
            )
            self._val_datasets = [
                ValDataset(name, transform=self.val_transform)
                for name in self.val_dataset_names
            ]

    def train_dataloader(self) -> DataLoader:
        batch_sampler = self._train_dataset.get_batch_sampler(
            self.batch_size, drop_last=True
        )
        return DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=TrainDataset.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader | List[DataLoader]:
        val_workers = max(1, self.num_workers // 2)
        return [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=val_workers,
                pin_memory=True,
                persistent_workers=val_workers > 0,
            )
            for ds in self._val_datasets
        ]
