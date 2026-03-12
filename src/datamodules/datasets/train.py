from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import pandas as pd


class PlaceImageTrainDataset:
    """Dataset indexed by place.  Each item returns M images sampled from one place.

    Parameters
    ----------
    index_path:
        Path to a parquet file with at least ``image_id``, ``image_path``,
        ``place_id``, and ``supergroup_id`` columns.
    images_per_place:
        Number of images (M) to sample for each place.  Places with fewer
        images than M are excluded.
    transform:
        Optional transform applied to each loaded image.
    seed:
        Base seed for image sampling.  Combined with the place index so
        sampling is deterministic but independent per place.
    """

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
        self._image_id_to_record = {
            str(row[self.image_id_column]): row
            for row in dataframe.to_dict("records")
        }
        self._places, self.valid_supergroup_ids = self._build_places(dataframe)

        if not self._places:
            raise ValueError(
                "No places found in the training index with a supergroup_id assigned."
            )

    def __len__(self) -> int:
        return len(self._places)

    def __getitem__(self, index: int) -> dict[str, Any]:
        place_id, supergroup_id, image_ids = self._places[index]
        sampled_ids = random.Random(self.seed + index).sample(image_ids, self.images_per_place)
        return {
            "place_id": place_id,
            "supergroup_id": supergroup_id,
            "image_ids": sampled_ids,
            "items": [self._build_image_item(self._image_id_to_record[iid]) for iid in sampled_ids],
        }

    def _build_places(
        self, dataframe: "pd.DataFrame"
    ) -> tuple[list[tuple[str, int, list[str]]], list[int]]:
        place_data: dict[str, tuple[int, list[str]]] = {}
        for record in dataframe.to_dict("records"):
            place_id = record.get(self.place_id_column)
            supergroup_id = record.get(self.supergroup_id_column)
            if place_id is None or supergroup_id is None:
                continue
            place_id = str(place_id)
            if place_id not in place_data:
                place_data[place_id] = (int(supergroup_id), [])
            place_data[place_id][1].append(str(record[self.image_id_column]))

        if place_data:
            min_images = min(len(image_ids) for _, image_ids in place_data.values())
            if self.images_per_place > min_images:
                raise ValueError(
                    f"images_per_place={self.images_per_place} exceeds the minimum number of "
                    f"images in any place ({min_images}). Reduce images_per_place or re-run "
                    f"the pipeline with a higher min_images_per_place."
                )

        places = []
        supergroup_ids = []
        for place_id, (supergroup_id, image_ids) in place_data.items():
            places.append((place_id, supergroup_id, image_ids))
            supergroup_ids.append(supergroup_id)

        return places, supergroup_ids

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
        required = (
            self.image_id_column,
            self.image_path_column,
            self.place_id_column,
            self.supergroup_id_column,
        )
        missing = [c for c in required if c not in dataframe.columns]
        if missing:
            raise KeyError(f"Training index is missing required columns: {', '.join(missing)}")

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
    """Yields batches of place indices where every place in a batch belongs to
    the same supergroup.

    Supergroups are visited one at a time.  Within each supergroup, places are
    chunked into batches of ``places_per_batch``.  Both supergroup order and
    place order within each supergroup are reshuffled every epoch via
    ``set_epoch()``.

    Parameters
    ----------
    supergroup_ids:
        One integer per dataset item (parallel to the dataset's places).
        Typically ``dataset.valid_supergroup_ids``.
    places_per_batch:
        Number of places (N) per batch.
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

        groups: dict[int, list[int]] = {}
        for idx, sg_id in enumerate(supergroup_ids):
            groups.setdefault(sg_id, []).append(idx)
        self._groups = groups

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self._epoch)

        supergroup_ids = list(self._groups.keys())
        if self.shuffle:
            rng.shuffle(supergroup_ids)

        for sg_id in supergroup_ids:
            indices = list(self._groups[sg_id])
            if self.shuffle:
                rng.shuffle(indices)
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
    """Flattens a batch of N place-dicts (each with M images) into stacked
    ``inputs`` and per-image ``labels`` (0..N-1, repeated M times each)."""
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
        supergroup_ids = {int(item["supergroup_id"]) for item in batch}
        if len(supergroup_ids) != 1:
            raise ValueError(
                "Train batch contained multiple supergroup_ids; "
                "SupergroupBatchSampler should keep each batch within one supergroup"
            )
        supergroup_id = supergroup_ids.pop()
        all_images: list = []
        all_labels: list = []
        for label, place in enumerate(batch):
            for img_item in place["items"]:
                all_images.append(_to_tensor(img_item["image"]))
                all_labels.append(label)
        return {
            "inputs": torch.stack(all_images),
            "labels": torch.tensor(all_labels),
            "supergroup_id": torch.tensor(supergroup_id),
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
