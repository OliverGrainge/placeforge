from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from PIL import Image

from datamodules.datamodule import PlaceRecognitionTrainDataModule


def _make_train_index(
    directory: Path, *, num_images_per_place: int = 4, num_places: int = 2
) -> None:
    """Write a minimal index.parquet under directory/."""
    rows = []
    for place_idx in range(num_places):
        for img_idx in range(num_images_per_place):
            image_id = f"place{place_idx}_img{img_idx}"
            image_path = directory / f"{image_id}.jpg"
            Image.new("RGB", (4, 4), color=(place_idx * 60, img_idx * 40, 0)).save(
                image_path
            )
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path),
                    "place_id": f"place_{place_idx}",
                    "supergroup_id": 0,
                }
            )
    pd.DataFrame(rows).to_parquet(directory / "index.parquet")


def _make_val_dataset(directory: Path) -> None:
    """Write a minimal val dataset (queries/database/matches/metadata) under directory/."""
    q_img = directory / "q0.jpg"
    db0_img = directory / "db0.jpg"
    db1_img = directory / "db1.jpg"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(q_img)
    Image.new("RGB", (4, 4), color=(0, 255, 0)).save(db0_img)
    Image.new("RGB", (4, 4), color=(0, 0, 255)).save(db1_img)

    pd.DataFrame(
        [{"qid": 0, "path": str(q_img), "lat": 37.77, "lon": -122.41}]
    ).to_parquet(directory / "queries.parquet")
    pd.DataFrame(
        [
            {"dbid": 0, "path": str(db0_img), "lat": 37.77, "lon": -122.41},
            {"dbid": 1, "path": str(db1_img), "lat": 37.78, "lon": -122.42},
        ]
    ).to_parquet(directory / "database.parquet")
    pd.DataFrame([{"qid": 0, "dbid": 0}]).to_parquet(directory / "matches.parquet")
    (directory / "metadata.json").write_text(
        json.dumps({"dataset_name": "test", "match_radius_m": 25.0}) + "\n"
    )


class PlaceRecognitionTrainDataModuleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)

        self.processed_dir = root / "processed"
        self.train_dir = self.processed_dir / "train" / "sf_xl_small_geometric"
        self.val_dir = self.processed_dir / "val" / "pitts30k"
        self.train_dir.mkdir(parents=True)
        self.val_dir.mkdir(parents=True)

        _make_train_index(self.train_dir)
        _make_val_dataset(self.val_dir)

        self._env_patch = mock.patch.dict(
            os.environ,
            {"PLACEFORGE_PROCESSED_DIR": str(self.processed_dir)},
            clear=False,
        )
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()
        self._tmp.cleanup()

    def _make_datamodule(self, **kwargs) -> PlaceRecognitionTrainDataModule:
        defaults = dict(
            train_dataset="sf_xl_small_geometric",
            places_per_batch=2,
            images_per_place=4,
            val_datasets=["pitts30k"],
        )
        defaults.update(kwargs)
        return PlaceRecognitionTrainDataModule(**defaults)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def test_setup_initializes_train_and_val_datasets(self) -> None:
        dm = self._make_datamodule()
        dm.setup("fit")

        self.assertIsNotNone(dm._train_dataset)
        self.assertEqual(len(dm._val_datasets), 1)
        self.assertEqual(dm._val_datasets[0].num_queries, 1)
        self.assertEqual(dm._val_datasets[0].num_database, 2)

    def test_setup_with_no_val_datasets(self) -> None:
        dm = self._make_datamodule(val_datasets=None)
        dm.setup("fit")

        self.assertIsNotNone(dm._train_dataset)
        self.assertEqual(dm._val_datasets, [])

    def test_setup_with_multiple_val_datasets(self) -> None:
        extra_val_dir = self.processed_dir / "val" / "tokyo247"
        extra_val_dir.mkdir(parents=True)
        _make_val_dataset(extra_val_dir)

        dm = self._make_datamodule(val_datasets=["pitts30k", "tokyo247"])
        dm.setup("fit")

        self.assertEqual(len(dm._val_datasets), 2)

    # ------------------------------------------------------------------
    # train_dataloader
    # ------------------------------------------------------------------

    def test_train_dataloader_produces_correct_batch_shape(self) -> None:
        dm = self._make_datamodule()
        batch = next(iter(dm.train_dataloader()))

        # 2 places × 4 images = 8 items per batch
        self.assertEqual(batch["inputs"].shape[0], 8)
        self.assertEqual(len(batch["labels"]), 8)

    def test_train_dataloader_labels_match_place_grouping(self) -> None:
        dm = self._make_datamodule()
        labels = next(iter(dm.train_dataloader()))["labels"].tolist()

        # Each place gets a unique label; within a group they are identical
        self.assertEqual(labels[:4], [labels[0]] * 4)
        self.assertEqual(labels[4:], [labels[4]] * 4)
        self.assertNotEqual(labels[0], labels[4])

    # ------------------------------------------------------------------
    # val_dataloader
    # ------------------------------------------------------------------

    def test_val_dataloader_returns_none_when_no_val_datasets(self) -> None:
        dm = self._make_datamodule(val_datasets=None)
        self.assertIsNone(dm.val_dataloader())

    def test_val_dataloader_returns_list_of_loaders(self) -> None:
        dm = self._make_datamodule()
        dm.setup("fit")
        loaders = dm.val_dataloader()

        self.assertIsInstance(loaders, list)
        self.assertEqual(len(loaders), 1)

    def test_val_dataloader_batch_covers_all_items(self) -> None:
        # batch_size = places_per_batch * images_per_place = 2 * 4 = 8
        # val dataset has 3 items (1 query + 2 db), so one batch covers all
        dm = self._make_datamodule()
        dm.setup("fit")
        batch = next(iter(dm.val_dataloader()[0]))

        self.assertEqual(batch["inputs"].shape[0], 3)
        self.assertEqual(batch["dataset_indices"].tolist(), [0, 1, 2])

    def test_val_dataloader_ground_truth(self) -> None:
        dm = self._make_datamodule()
        dm.setup("fit")
        loader = dm.val_dataloader()[0]

        # query at dataset index 2 (after 2 db images), positive is db 0
        self.assertEqual(loader.dataset.ground_truth(), [(2, [0])])

    def test_val_dataloader_multiple_datasets(self) -> None:
        extra_val_dir = self.processed_dir / "val" / "tokyo247"
        extra_val_dir.mkdir(parents=True)
        _make_val_dataset(extra_val_dir)

        dm = self._make_datamodule(val_datasets=["pitts30k", "tokyo247"])
        dm.setup("fit")
        loaders = dm.val_dataloader()

        self.assertEqual(len(loaders), 2)
        for loader in loaders:
            self.assertEqual(loader.dataset.num_queries, 1)
            self.assertEqual(loader.dataset.num_database, 2)

    # ------------------------------------------------------------------
    # error handling
    # ------------------------------------------------------------------

    def test_missing_train_dataset_raises_file_not_found(self) -> None:
        dm = self._make_datamodule(train_dataset="nonexistent_dataset")
        with self.assertRaises(FileNotFoundError):
            dm.setup("fit")

    def test_missing_val_dataset_raises_file_not_found(self) -> None:
        dm = self._make_datamodule(val_datasets=["pitts30k", "nonexistent_val"])
        with self.assertRaises(FileNotFoundError):
            dm.setup("fit")

    # ------------------------------------------------------------------
    # transforms and batch_size
    # ------------------------------------------------------------------

    def test_val_transform_defaults_to_train_transform(self) -> None:
        transform = object()
        dm = self._make_datamodule(train_transform=transform)
        self.assertIs(dm.val_transform, transform)

    def test_val_transform_can_be_set_independently(self) -> None:
        train_t, val_t = object(), object()
        dm = self._make_datamodule(train_transform=train_t, val_transform=val_t)
        self.assertIs(dm.train_transform, train_t)
        self.assertIs(dm.val_transform, val_t)

    def test_batch_size_is_places_times_images(self) -> None:
        dm = self._make_datamodule(places_per_batch=3, images_per_place=5)
        self.assertEqual(dm.batch_size, 15)


if __name__ == "__main__":
    unittest.main()
