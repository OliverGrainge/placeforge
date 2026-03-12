from __future__ import annotations

from pathlib import Path
from typing import Any

from . import register_datamodule
from .datasets.train import PlaceImageTrainDataset, build_train_dataloader

try:
    import lightning as L
except ModuleNotFoundError:
    try:
        import pytorch_lightning as L
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PlaceRecognitionTrainDataModule requires `lightning` or `pytorch-lightning`."
        ) from exc


@register_datamodule("place_recognition_train")
class PlaceRecognitionTrainDataModule(L.LightningDataModule):
    def __init__(
        self,
        index_path: str | Path,
        *,
        places_per_batch: int,
        images_per_place: int,
        transform: Any = None,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        load_images: bool = True,
    ) -> None:
        super().__init__()
        self.index_path = Path(index_path)
        self.places_per_batch = places_per_batch
        self.images_per_place = images_per_place
        self.transform = transform
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.load_images = load_images
        self.train_dataset: PlaceImageTrainDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = PlaceImageTrainDataset(
                self.index_path,
                images_per_place=self.images_per_place,
                transform=self.transform,
                load_images=self.load_images,
                seed=self.seed,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            self.setup("fit")

        if self.train_dataset is None:
            raise RuntimeError("Training dataset was not initialized")

        return build_train_dataloader(
            self.index_path,
            places_per_batch=self.places_per_batch,
            images_per_place=self.images_per_place,
            transform=self.transform,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
            seed=self.seed,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            load_images=self.load_images,
            dataset=self.train_dataset,
        )


__all__ = ["PlaceRecognitionTrainDataModule"]
