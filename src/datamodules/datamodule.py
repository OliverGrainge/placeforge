from __future__ import annotations

import os
from typing import Any, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .datasets import EvalDataset


def _dataloader_worker_init_fn(_worker_id: int) -> None:
    """Avoid CPU thread oversubscription inside dataloader workers.

    Image decode and torchvision tensor transforms can use PyTorch CPU thread
    pools. With multiple workers, leaving the per-worker thread count at the
    process default causes heavy contention and periodic stalls.
    """
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        torch.set_num_interop_threads(1)


class BaseDataModule(pl.LightningDataModule):
    """Base datamodule providing validation and test dataloader logic.

    Subclass this and implement ``setup_train`` / ``train_dataloader`` to
    define the training behaviour.
    """

    def __init__(
        self,
        val_dataset_names: List[str],
        batch_size: int,
        num_workers: int = os.cpu_count() // 2,
        val_transform: Any = None,
        test_dataset_names: List[str] | None = None,
        test_transform: Any = None,
    ):
        super().__init__()
        self.val_dataset_names = val_dataset_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_transform = val_transform
        self.test_dataset_names = test_dataset_names or []
        self.test_transform = test_transform

        self._val_datasets: List[EvalDataset] = []
        self._test_datasets: List[EvalDataset] = []

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._val_datasets = [
                EvalDataset(name, transform=self.val_transform)
                for name in self.val_dataset_names
            ]
        if stage in (None, "test"):
            self._test_datasets = [
                EvalDataset(name, transform=self.test_transform)
                for name in self.test_dataset_names
            ]

    def _eval_dataloader(self, ds: EvalDataset) -> DataLoader:
        workers = max(1, self.num_workers // 2)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=workers,
            worker_init_fn=_dataloader_worker_init_fn if workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )

    def val_dataloader(self) -> List[DataLoader]:
        return [self._eval_dataloader(ds) for ds in self._val_datasets]

    def test_dataloader(self) -> List[DataLoader]:
        return [self._eval_dataloader(ds) for ds in self._test_datasets]
