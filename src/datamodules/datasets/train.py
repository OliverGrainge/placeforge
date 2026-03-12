from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import pandas as pd


class PlaceImageTrainDataset:
    def __init__(
        self,
        index_path: str | Path,
        *,
        images_per_place: int,
        transform: Any = None,
        image_id_column: str = "image_id",
        image_path_column: str = "image_path",
        place_id_column: str = "place_id",
        supergroup_id_column: str = "supergroup_id",
        load_images: bool = True,
        seed: int = 0,
    ) -> None:
        if images_per_place <= 0:
            raise ValueError("images_per_place must be greater than 0")

        self.index_path = Path(index_path)
        self.images_per_place = images_per_place
        self.transform = transform
        self.image_id_column = image_id_column
        self.image_path_column = image_path_column
        self.place_id_column = place_id_column
        self.supergroup_id_column = supergroup_id_column
        self.load_images = load_images
        self.seed = seed

        dataframe = self._read_index()
        self._validate_columns(dataframe)
        self.records = dataframe.to_dict("records")
        self.image_id_to_record = {
            str(record[self.image_id_column]): record for record in self.records
        }
        self.place_id_to_image_ids = self._build_place_index()
        self.valid_indices, self.valid_supergroup_ids = self._build_valid_indices()

        if not self.valid_indices:
            raise ValueError(
                "No training rows have enough same-place images for the requested "
                "images_per_place value, or no rows have a supergroup_id assigned"
            )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        anchor_index = self.valid_indices[index]
        anchor_record = self.records[anchor_index]
        anchor_image_id = str(anchor_record[self.image_id_column])
        positive_image_ids = self._sample_positive_image_ids(anchor_record, seed_offset=index)
        group_image_ids = [anchor_image_id, *positive_image_ids]
        group_records = [self.image_id_to_record[image_id] for image_id in group_image_ids]

        items = [self._build_image_item(record) for record in group_records]
        return {
            "anchor_index": anchor_index,
            "anchor_image_id": anchor_image_id,
            "image_ids": group_image_ids,
            "items": items,
        }

    def _build_valid_indices(self) -> tuple[list[int], list[int]]:
        required_positive_count = self.images_per_place - 1
        valid_indices: list[int] = []
        valid_supergroup_ids: list[int] = []

        for index, record in enumerate(self.records):
            supergroup_id = record.get(self.supergroup_id_column)
            if supergroup_id is None:
                continue
            positive_image_ids = self._positive_image_ids(record)
            if len(positive_image_ids) >= required_positive_count:
                valid_indices.append(index)
                valid_supergroup_ids.append(int(supergroup_id))

        return valid_indices, valid_supergroup_ids

    def _sample_positive_image_ids(self, anchor_record: dict[str, Any], *, seed_offset: int) -> list[str]:
        required_positive_count = self.images_per_place - 1
        if required_positive_count == 0:
            return []

        matches = self._positive_image_ids(anchor_record)
        generator = random.Random(self.seed + seed_offset)
        return generator.sample(matches, required_positive_count)

    def _build_place_index(self) -> dict[str, list[str]]:
        place_id_to_image_ids: dict[str, list[str]] = {}
        for record in self.records:
            place_id = record.get(self.place_id_column)
            if place_id is None:
                continue
            image_id = str(record[self.image_id_column])
            place_id_to_image_ids.setdefault(str(place_id), []).append(image_id)
        return place_id_to_image_ids

    def _positive_image_ids(self, record: dict[str, Any]) -> list[str]:
        place_id = record.get(self.place_id_column)
        if place_id is None:
            return []

        raw_matches = self.place_id_to_image_ids.get(str(place_id), [])
        anchor_image_id = str(record[self.image_id_column])
        matches: list[str] = []
        seen: set[str] = set()

        for match_image_id in raw_matches:
            image_id = str(match_image_id)
            if image_id == anchor_image_id or image_id in seen:
                continue
            if image_id not in self.image_id_to_record:
                continue
            seen.add(image_id)
            matches.append(image_id)

        return matches

    def _build_image_item(self, record: dict[str, Any]) -> dict[str, Any]:
        item = {
            "image_id": str(record[self.image_id_column]),
            "image_path": str(record[self.image_path_column]),
        }

        if self.load_images:
            image = self._load_image(item["image_path"])
            if self.transform is not None:
                image = self.transform(image)
            item["image"] = image

        for key, value in record.items():
            if key not in item:
                item[key] = value

        return item

    def _read_index(self) -> "pd.DataFrame":
        pd = self._import_pandas()
        if not self.index_path.exists():
            raise FileNotFoundError(f"Training index does not exist: {self.index_path}")
        return pd.read_parquet(self.index_path)

    def _validate_columns(self, dataframe: "pd.DataFrame") -> None:
        required_columns = (
            self.image_id_column,
            self.image_path_column,
            self.place_id_column,
            self.supergroup_id_column,
        )
        missing_columns = [column for column in required_columns if column not in dataframe.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise KeyError(f"Training index is missing required columns: {missing}")

    @staticmethod
    def _import_pandas():
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PlaceImageTrainDataset requires pandas. Install it with `pip install pandas`."
            ) from exc

        return pd

    def _load_image(self, image_path: str) -> Any:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading images requires Pillow. Install it with `pip install pillow`."
            ) from exc

        with Image.open(image_path) as image:
            return image.convert("RGB")


class SupergroupBatchSampler:
    """Yields batches of dataset indices where every index in a batch belongs
    to the same supergroup.

    Supergroups are iterated one at a time — all batches from supergroup A are
    yielded before moving on to supergroup B.  This guarantees that any N
    places drawn in a single batch are mutually non-adjacent geographically,
    making them safe hard negatives for metric learning.

    Both the supergroup order and the place order within each supergroup are
    re-shuffled every epoch via ``set_epoch()``.

    Parameters
    ----------
    supergroup_ids:
        One integer per dataset item (parallel to the dataset's valid_indices).
        Typically ``dataset.valid_supergroup_ids``.
    places_per_batch:
        Number of places (N) per batch.  Each place contributes
        ``images_per_place`` images, so the effective batch size seen by the
        model is ``N × M``.
    shuffle:
        Shuffle supergroup order and place order within each supergroup.
    drop_last:
        Drop the final incomplete batch of each supergroup if it has fewer
        than ``places_per_batch`` places.
    seed:
        Base random seed; the actual seed per epoch is ``seed + epoch``.
    """

    def __init__(
        self,
        supergroup_ids: list[int],
        places_per_batch: int,
        *,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        if places_per_batch <= 0:
            raise ValueError("places_per_batch must be positive")

        self.places_per_batch = places_per_batch
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._epoch = 0

        # Build a mapping from supergroup_id → list of dataset indices
        groups: dict[int, list[int]] = {}
        for idx, sg_id in enumerate(supergroup_ids):
            if sg_id not in groups:
                groups[sg_id] = []
            groups[sg_id].append(idx)
        self._groups = groups

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch to get a fresh shuffle."""
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)

        # Determine the order in which to visit supergroups
        supergroup_ids = list(self._groups.keys())
        if self.shuffle:
            rng.shuffle(supergroup_ids)

        for sg_id in supergroup_ids:
            indices = list(self._groups[sg_id])
            if self.shuffle:
                rng.shuffle(indices)

            # Yield fixed-size batches from this supergroup before moving on
            for start in range(0, len(indices), self.places_per_batch):
                batch = indices[start : start + self.places_per_batch]
                if len(batch) < self.places_per_batch and self.drop_last:
                    continue
                yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self._groups.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.places_per_batch
            else:
                total += (n + self.places_per_batch - 1) // self.places_per_batch
        return total


def _make_collate_fn():
    """Returns a collate function that flattens the nested items structure into
    ``inputs`` (stacked image tensors) and ``labels`` (per-image place index)."""
    import torch

    def _to_tensor(image: Any) -> "torch.Tensor":
        if isinstance(image, torch.Tensor):
            return image
        try:
            from torchvision.transforms.functional import to_tensor as tvf_to_tensor
            return tvf_to_tensor(image)
        except ImportError:
            import numpy as np
            arr = np.array(image)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            return torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0

    def collate_fn(batch: list[dict]) -> dict:
        all_images: list = []
        all_labels: list = []
        for label, item in enumerate(batch):
            for img_item in item["items"]:
                all_images.append(_to_tensor(img_item["image"]))
                all_labels.append(label)
        return {
            "inputs": torch.stack(all_images),
            "labels": torch.tensor(all_labels),
        }

    return collate_fn


def build_train_dataloader(
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
    dataset: PlaceImageTrainDataset | None = None,
):
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Building a train dataloader requires torch. Install it with `pip install torch`."
        ) from exc

    train_dataset = dataset or PlaceImageTrainDataset(
        index_path,
        images_per_place=images_per_place,
        transform=transform,
        load_images=load_images,
        seed=seed,
    )

    batch_sampler = SupergroupBatchSampler(
        train_dataset.valid_supergroup_ids,
        places_per_batch=places_per_batch,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )

    # batch_sampler takes over batch composition, so batch_size/shuffle/drop_last
    # must not be passed separately to DataLoader
    return DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_make_collate_fn(),
    )


__all__ = [
    "PlaceImageTrainDataset",
    "SupergroupBatchSampler",
    "build_train_dataloader",
]
