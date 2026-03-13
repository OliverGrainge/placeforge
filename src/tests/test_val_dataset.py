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
from datamodules.datasets.val import VPRValidationDataset, build_validation_dataloader


class VPRValidationDatasetTests(unittest.TestCase):
    def test_loads_vpr_validation_dataset_and_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "processed" / "pitts30k"
            image_root = root / "raw" / "pitts30k" / "images" / "val"
            queries_dir = image_root / "queries"
            database_dir = image_root / "database"
            queries_dir.mkdir(parents=True)
            database_dir.mkdir(parents=True)

            Image.new("RGB", (4, 4), color=(255, 0, 0)).save(queries_dir / "q0.jpg")
            Image.new("RGB", (4, 4), color=(0, 255, 0)).save(database_dir / "db0.jpg")
            Image.new("RGB", (4, 4), color=(0, 0, 255)).save(database_dir / "db1.jpg")

            dataset_dir.mkdir(parents=True)
            pd.DataFrame(
                [{"qid": 0, "path": "queries/q0.jpg", "lat": 1.0, "lon": 2.0}]
            ).to_parquet(dataset_dir / "queries.parquet")
            pd.DataFrame(
                [
                    {"dbid": 0, "path": "database/db0.jpg", "lat": 3.0, "lon": 4.0},
                    {"dbid": 1, "path": "database/db1.jpg", "lat": 5.0, "lon": 6.0},
                ]
            ).to_parquet(dataset_dir / "database.parquet")
            pd.DataFrame([{"qid": 0, "dbid": 1}]).to_parquet(dataset_dir / "matches.parquet")
            (dataset_dir / "metadata.json").write_text(
                json.dumps({"dataset_name": "pitts30k", "version": "1.0"}) + "\n"
            )

            dataset = VPRValidationDataset(dataset_dir, image_root=image_root, load_images=False)

            self.assertEqual(dataset.num_database, 2)
            self.assertEqual(dataset.num_queries, 1)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0]["dataset_index"], 0)
            self.assertEqual(dataset[2]["dataset_index"], 2)
            self.assertEqual(dataset[2]["positive_dbids"], [1])
            self.assertEqual(dataset.ground_truth(), [(2, [1])])
            self.assertEqual(
                dataset[2]["image_path"],
                str(image_root / "queries" / "q0.jpg"),
            )

    def test_validation_dataloader_collates_metric_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            dataset_dir = root / "dataset"
            images_dir = root / "images"
            (images_dir / "queries").mkdir(parents=True)
            (images_dir / "database").mkdir(parents=True)

            for rel_path, color in (
                ("queries/q0.jpg", (255, 0, 0)),
                ("database/db0.jpg", (0, 255, 0)),
                ("database/db1.jpg", (0, 0, 255)),
            ):
                Image.new("RGB", (3, 3), color=color).save(images_dir / rel_path)

            dataset_dir.mkdir(parents=True)
            pd.DataFrame(
                [{"qid": 0, "path": "queries/q0.jpg", "lat": 1.0, "lon": 2.0}]
            ).to_parquet(dataset_dir / "queries.parquet")
            pd.DataFrame(
                [
                    {"dbid": 0, "path": "database/db0.jpg", "lat": 3.0, "lon": 4.0},
                    {"dbid": 1, "path": "database/db1.jpg", "lat": 5.0, "lon": 6.0},
                ]
            ).to_parquet(dataset_dir / "database.parquet")
            pd.DataFrame([{"qid": 0, "dbid": 0}, {"qid": 0, "dbid": 1}]).to_parquet(
                dataset_dir / "matches.parquet"
            )
            (dataset_dir / "metadata.json").write_text(json.dumps({"dataset_name": "test"}) + "\n")

            dataloader = build_validation_dataloader(
                dataset_dir,
                batch_size=3,
                image_root=images_dir,
            )
            batch = next(iter(dataloader))

            self.assertEqual(batch["inputs"].shape[0], 3)
            self.assertEqual(batch["dataset_indices"].tolist(), [0, 1, 2])
            self.assertEqual(dataloader.dataset.ground_truth(), [(2, [0, 1])])

    def test_dataset_resolves_raw_relative_paths_from_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_root = root / "raw"
            dataset_dir = root / "processed" / "val" / "pitts30k"
            image_path = raw_root / "pitts30k" / "images" / "val" / "database" / "db0.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (3, 3), color=(0, 255, 0)).save(image_path)

            dataset_dir.mkdir(parents=True)
            pd.DataFrame([], columns=["qid", "path", "lat", "lon"]).to_parquet(
                dataset_dir / "queries.parquet"
            )
            pd.DataFrame(
                [
                    {
                        "dbid": 0,
                        "path": "pitts30k/images/val/database/db0.jpg",
                        "lat": 3.0,
                        "lon": 4.0,
                    }
                ]
            ).to_parquet(dataset_dir / "database.parquet")
            pd.DataFrame([], columns=["qid", "dbid"]).to_parquet(dataset_dir / "matches.parquet")
            (dataset_dir / "metadata.json").write_text(json.dumps({"dataset_name": "pitts30k"}) + "\n")

            with mock.patch.dict(os.environ, {"PLACEFORGE_RAW_DIR": str(raw_root)}, clear=False):
                dataset = VPRValidationDataset(dataset_dir, load_images=False)

            self.assertEqual(dataset[0]["image_path"], str(image_path))

    def test_train_datamodule_exposes_validation_dataloader(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            processed = root / "processed"
            train_dir = processed / "train" / "my_dataset"
            val_dir = processed / "val" / "my_val"
            train_dir.mkdir(parents=True)
            val_dir.mkdir(parents=True)

            q_img = root / "q0.jpg"
            db_img = root / "db0.jpg"
            Image.new("RGB", (3, 3), color=(255, 0, 0)).save(q_img)
            Image.new("RGB", (3, 3), color=(0, 255, 0)).save(db_img)

            pd.DataFrame(
                [
                    {"image_id": "a1", "image_path": str(root / "a1.jpg"), "place_id": "place_a", "supergroup_id": 0},
                    {"image_id": "a2", "image_path": str(root / "a2.jpg"), "place_id": "place_a", "supergroup_id": 0},
                ]
            ).to_parquet(train_dir / "index.parquet")

            pd.DataFrame([{"qid": 0, "path": str(q_img), "lat": 1.0, "lon": 2.0}]).to_parquet(val_dir / "queries.parquet")
            pd.DataFrame([{"dbid": 0, "path": str(db_img), "lat": 3.0, "lon": 4.0}]).to_parquet(val_dir / "database.parquet")
            pd.DataFrame([{"qid": 0, "dbid": 0}]).to_parquet(val_dir / "matches.parquet")
            (val_dir / "metadata.json").write_text(json.dumps({"dataset_name": "test"}) + "\n")

            with mock.patch.dict(os.environ, {"PLACEFORGE_PROCESSED_DIR": str(processed)}, clear=False):
                datamodule = PlaceRecognitionTrainDataModule(
                    "my_dataset",
                    places_per_batch=1,
                    images_per_place=2,
                    val_datasets=["my_val"],
                )
                datamodule.setup("fit")
                val_loader = datamodule.val_dataloader()[0]

            batch = next(iter(val_loader))

            self.assertEqual(batch["inputs"].shape[0], 2)
            self.assertEqual(batch["dataset_indices"].tolist(), [0, 1])
            self.assertEqual(val_loader.dataset.ground_truth(), [(1, [0])])


if __name__ == "__main__":
    unittest.main()
