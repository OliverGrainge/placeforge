from __future__ import annotations

import os
from typing import Any, List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from . import register_datamodule
from .datasets import ContrastiveTrainDataset, ClassificationTrainDataset, ValDataset, TestDataset
from .datasets.graded import GradedSimilarityTrainDataset


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

        self._val_datasets: List[ValDataset] = []
        self._test_datasets: List[TestDataset] = []

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._val_datasets = [
                ValDataset(name, transform=self.val_transform)
                for name in self.val_dataset_names
            ]
        if stage in (None, "test"):
            self._test_datasets = []
            for name in self.test_dataset_names:
                try:
                    self._test_datasets.append(TestDataset(name, transform=self.test_transform))
                except FileNotFoundError:
                    self._test_datasets.append(ValDataset(name, transform=self.test_transform))

    def _eval_dataloader(self, ds: ValDataset | TestDataset) -> DataLoader:
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


@register_datamodule("place_recognition")
class PlaceRecognitionTrainDataModule(BaseDataModule):
    def __init__(
        self,
        train_dataset_name: str,
        val_dataset_names: List[str],
        batch_size: int,
        num_workers: int = os.cpu_count() // 2,
        images_per_place: int = 4,
        train_transform: Any = None,
        val_transform: Any = None,
        test_dataset_names: List[str] | None = None,
        test_transform: Any = None,
        dataset_type: str = "contrastive",
        sampler: dict | None = None,
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
        self.train_transform = train_transform
        if dataset_type not in ("contrastive", "classification"):
            raise ValueError(f"dataset_type must be 'contrastive' or 'classification', got '{dataset_type}'")
        self.dataset_type = dataset_type
        self._sampler = sampler or {}
        self.save_hyperparameters()

        self._train_dataset: ContrastiveTrainDataset | ClassificationTrainDataset | None = None

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
            sampler_kwargs = {
                "num_supergroups": self._sampler.get("num_supergroups"),
                "num_places": self._sampler.get("num_places"),
                "min_places_per_supergroup": self._sampler.get("min_places_per_supergroup"),
                "order_by": self._sampler.get("order_by"),
            }
            if self.dataset_type == "classification":
                self._train_dataset = ClassificationTrainDataset(
                    self.train_dataset_name,
                    transform=self.train_transform,
                    **sampler_kwargs,
                )
            else:
                self._train_dataset = ContrastiveTrainDataset(
                    self.train_dataset_name,
                    images_per_place=self.images_per_place,
                    transform=self.train_transform,
                    **sampler_kwargs,
                )
        super().setup(stage)

    def train_dataloader(self) -> DataLoader:
        use_batch_sampler = isinstance(self._train_dataset, ContrastiveTrainDataset)
        if use_batch_sampler:
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
                in_order=False,
            )
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self._train_dataset.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=_dataloader_worker_init_fn if self.num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )


@register_datamodule("graded_similarity")
class GradedSimilarityTrainDataModule(BaseDataModule):
    def __init__(
        self,
        train_dataset_name: str,
        val_dataset_names: List[str],
        batch_size: int,
        num_workers: int = os.cpu_count() // 2,
        train_transform: Any = None,
        val_transform: Any = None,
        test_dataset_names: List[str] | None = None,
        test_transform: Any = None,
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
        self.train_transform = train_transform
        self.save_hyperparameters()

        self._train_dataset: GradedSimilarityTrainDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._train_dataset = GradedSimilarityTrainDataset(
                self.train_dataset_name,
                transform=self.train_transform,
            )
        super().setup(stage)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=GradedSimilarityTrainDataset.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=_dataloader_worker_init_fn if self.num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
