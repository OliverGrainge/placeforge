from __future__ import annotations

from pathlib import Path
from typing import Any

from . import register_datamodule
from .datasets.train import PlaceImageTrainDataset, build_train_dataloader
from .datasets.val import VPRValidationDataset, build_validation_dataloader

try:
    import lightning as L
except ModuleNotFoundError:
    try:
        import pytorch_lightning as L
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PlaceRecognitionTrainDataModule requires `lightning` or `pytorch-lightning`."
        ) from exc


def _processed_dir() -> Path:
    from datapipelines.env import processed_dir
    return processed_dir()


def _resolve_train_index(name: str) -> Path:
    path = _processed_dir() / "train" / name / "index.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Train dataset {name!r} not found. Expected index at: {path}"
        )
    return path


def _resolve_val_dir(name: str) -> Path:
    path = _processed_dir() / "val" / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"Val dataset {name!r} not found. Expected directory at: {path}"
        )
    return path


@register_datamodule("place_recognition_train")
class PlaceRecognitionTrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset: str,
        *,
        places_per_batch: int,
        images_per_place: int,
        val_datasets: list[str] | None = None,
        train_transform: Any = None,
        val_transform: Any = None,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()
        self.train_dataset_name = train_dataset
        self.val_dataset_names = val_datasets or []
        self.places_per_batch = places_per_batch
        self.images_per_place = images_per_place
        self.train_transform = train_transform
        self.val_transform = val_transform if val_transform is not None else train_transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._train_dataset: PlaceImageTrainDataset | None = None
        self._val_datasets: list[VPRValidationDataset] = []

    @property
    def batch_size(self) -> int:
        return self.places_per_batch * self.images_per_place

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._train_dataset = PlaceImageTrainDataset(
                _resolve_train_index(self.train_dataset_name),
                images_per_place=self.images_per_place,
                transform=self.train_transform,
                seed=self.seed,
            )
            self._val_datasets = [
                VPRValidationDataset(
                    _resolve_val_dir(name),
                    transform=self.val_transform,
                )
                for name in self.val_dataset_names
            ]

    def train_dataloader(self):
        if self._train_dataset is None:
            self.setup("fit")

        if self._train_dataset is None:
            raise RuntimeError("Training dataset was not initialized")

        return build_train_dataloader(
            _resolve_train_index(self.train_dataset_name),
            places_per_batch=self.places_per_batch,
            images_per_place=self.images_per_place,
            transform=self.train_transform,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            seed=self.seed,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            dataset=self._train_dataset,
        )

    def val_dataloader(self):
        if not self.val_dataset_names:
            return None

        if not self._val_datasets:
            self.setup("fit")

        return [
            build_validation_dataloader(
                _resolve_val_dir(name),
                batch_size=self.batch_size,
                transform=self.val_transform,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                dataset=ds,
            )
            for name, ds in zip(self.val_dataset_names, self._val_datasets)
        ]


__all__ = ["PlaceRecognitionTrainDataModule"]
