from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from datamodules.datasets.train import PlaceImageTrainDataset, build_train_dataloader


def _make_index(tmp_dir: Path, rows: list[dict], create_images: bool = False) -> Path:
    if create_images:
        for row in rows:
            path = Path(row["image_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 4), color=(0, 0, 0)).save(path)
    index_path = tmp_dir / "index.parquet"
    pd.DataFrame(rows).to_parquet(index_path)
    return index_path


class PlaceImageTrainDatasetTests(unittest.TestCase):
    def test_indexed_by_place(self) -> None:
        rows = [
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
            {
                "image_id": "b2",
                "image_path": "/tmp/b2.jpg",
                "place_id": "place_b",
                "supergroup_id": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = _make_index(Path(tmp_dir), rows)
            dataset = PlaceImageTrainDataset(
                index_path, images_per_place=2, load_images=False, seed=0
            )

        self.assertEqual(len(dataset), 2)  # two places
        self.assertEqual(dataset.valid_supergroup_ids, [0, 1])

    def test_getitem_returns_m_images_from_place(self) -> None:
        rows = [
            {
                "image_id": f"a{i}",
                "image_path": f"/tmp/a{i}.jpg",
                "place_id": "place_a",
                "supergroup_id": 0,
            }
            for i in range(5)
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = _make_index(Path(tmp_dir), rows)
            dataset = PlaceImageTrainDataset(
                index_path, images_per_place=3, load_images=False, seed=0
            )
            item = dataset[0]

        self.assertEqual(item["place_id"], "place_a")
        self.assertEqual(item["supergroup_id"], 0)
        self.assertEqual(len(item["image_ids"]), 3)
        self.assertEqual(len(item["items"]), 3)
        self.assertTrue(all(iid.startswith("a") for iid in item["image_ids"]))

    def test_raises_when_images_per_place_exceeds_minimum(self) -> None:
        rows = [
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
                "image_id": "b1",
                "image_path": "/tmp/b1.jpg",
                "place_id": "place_b",
                "supergroup_id": 0,
            },
            {
                "image_id": "b2",
                "image_path": "/tmp/b2.jpg",
                "place_id": "place_b",
                "supergroup_id": 0,
            },
            {
                "image_id": "b3",
                "image_path": "/tmp/b3.jpg",
                "place_id": "place_b",
                "supergroup_id": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = _make_index(Path(tmp_dir), rows)
            # place_a has 2 images, place_b has 3 — requesting 3 exceeds place_a's count
            with self.assertRaisesRegex(ValueError, "images_per_place=3 exceeds"):
                PlaceImageTrainDataset(
                    index_path, images_per_place=3, load_images=False
                )

    def test_dataloader_batch_shape_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            rows = []
            for place_idx in range(3):
                for img_idx in range(4):
                    image_id = f"p{place_idx}_{img_idx}"
                    image_path = tmp / f"{image_id}.jpg"
                    Image.new("RGB", (4, 4), color=(place_idx * 60, 0, 0)).save(
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

            index_path = _make_index(tmp, rows)
            dataset = PlaceImageTrainDataset(
                index_path, images_per_place=2, load_images=True, seed=0
            )
            loader = build_train_dataloader(
                index_path,
                places_per_batch=2,
                images_per_place=2,
                shuffle=False,
                drop_last=True,
                load_images=True,
                dataset=dataset,
            )
            batch = next(iter(loader))

        # 2 places × 2 images = 4 inputs
        self.assertEqual(batch["inputs"].shape[0], 4)
        # labels: [0, 0, 1, 1]
        self.assertEqual(batch["labels"].tolist(), [0, 0, 1, 1])
        self.assertEqual(int(batch["supergroup_id"]), 0)

    def test_sampling_is_deterministic(self) -> None:
        rows = [
            {
                "image_id": f"a{i}",
                "image_path": f"/tmp/a{i}.jpg",
                "place_id": "place_a",
                "supergroup_id": 0,
            }
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_path = _make_index(Path(tmp_dir), rows)
            ds1 = PlaceImageTrainDataset(
                index_path, images_per_place=3, load_images=False, seed=42
            )
            ds2 = PlaceImageTrainDataset(
                index_path, images_per_place=3, load_images=False, seed=42
            )

        self.assertEqual(ds1[0]["image_ids"], ds2[0]["image_ids"])


if __name__ == "__main__":
    unittest.main()
