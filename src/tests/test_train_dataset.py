from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from datamodules.datasets.train import PlaceImageTrainDataset


class PlaceImageTrainDatasetTests(unittest.TestCase):
    def test_uses_place_ids_to_find_positives(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "image_id": "a1",
                    "image_path": "/tmp/a1.jpg",
                    "place_id": "place_a",
                    "supergroup_id": 0,
                },
                {
                    "image_id": "a2",
                    "image_path": "/tmp/a2.jpg",
                    "place_id": "place_a",
                    "supergroup_id": 0,
                },
                {
                    "image_id": "a3",
                    "image_path": "/tmp/a3.jpg",
                    "place_id": "place_a",
                    "supergroup_id": 0,
                },
                {
                    "image_id": "b1",
                    "image_path": "/tmp/b1.jpg",
                    "place_id": "place_b",
                    "supergroup_id": 1,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "index.parquet"
            dataframe.to_parquet(index_path)

            dataset = PlaceImageTrainDataset(index_path, images_per_place=3, load_images=False, seed=0)

            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.valid_supergroup_ids, [0, 0, 0])

            sample = dataset[0]

        self.assertEqual(sample["anchor_image_id"], "a1")
        self.assertEqual(len(sample["image_ids"]), 3)
        self.assertEqual(set(sample["image_ids"]), {"a1", "a2", "a3"})

    def test_rejects_places_smaller_than_images_per_place(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "image_id": "a1",
                    "image_path": "/tmp/a1.jpg",
                    "place_id": "place_a",
                    "supergroup_id": 0,
                },
                {
                    "image_id": "a2",
                    "image_path": "/tmp/a2.jpg",
                    "place_id": "place_a",
                    "supergroup_id": 0,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = Path(tmp_dir) / "index.parquet"
            dataframe.to_parquet(index_path)

            with self.assertRaisesRegex(ValueError, "same-place images"):
                PlaceImageTrainDataset(index_path, images_per_place=3, load_images=False)


if __name__ == "__main__":
    unittest.main()
