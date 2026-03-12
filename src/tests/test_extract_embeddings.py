from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from datapipelines.steps.extractembeddings import ExtractEmbeddingsStep


class ExtractEmbeddingsStepTests(unittest.TestCase):
    def test_uses_context_override_for_cache_dir_and_exposes_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "custom_cache"
            dataframe = pd.DataFrame(
                [
                    {"image_id": "img_1", "image_path": "/tmp/one.jpg"},
                    {"image_id": "img_2", "image_path": "/tmp/two.jpg"},
                ]
            )
            step = ExtractEmbeddingsStep(
                cache_dir=Path(tmp_dir) / "default_cache",
                batch_size=2,
                cache_dir_context_key="embedding_cache_dir",
            )

            with patch.object(step, "_resolve_device", return_value="cpu"), patch.object(
                step, "_load_model", return_value=object()
            ), patch.object(step, "_build_transform", return_value=object()), patch.object(
                step,
                "_embed_batch",
                return_value=np.ones((2, step.embed_dim), dtype=np.float32),
            ):
                result = step.run({"index": dataframe, "embedding_cache_dir": cache_dir})

            self.assertEqual(result["embedding_cache_dir"], cache_dir)
            self.assertEqual(
                result["embedding_index_path"],
                cache_dir / "manifest.parquet",
            )
            self.assertTrue((cache_dir / "batches" / "batch_000000.npy").exists())
            self.assertTrue((cache_dir / "manifest.parquet").exists())

    def test_resumes_after_interrupted_run_from_last_flushed_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "embedding_cache"
            dataframe = pd.DataFrame(
                [
                    {"image_id": "img_1", "image_path": "/tmp/one.jpg"},
                    {"image_id": "img_2", "image_path": "/tmp/two.jpg"},
                    {"image_id": "img_3", "image_path": "/tmp/three.jpg"},
                    {"image_id": "img_4", "image_path": "/tmp/four.jpg"},
                ]
            )
            step = ExtractEmbeddingsStep(cache_dir=cache_dir, batch_size=2)
            first_batch = np.full((2, step.embed_dim), 1.0, dtype=np.float32)
            second_batch = np.full((2, step.embed_dim), 2.0, dtype=np.float32)

            with patch.object(step, "_resolve_device", return_value="cpu"), patch.object(
                step, "_load_model", return_value=object()
            ), patch.object(step, "_build_transform", return_value=object()), patch.object(
                step,
                "_embed_batch",
                side_effect=[first_batch, RuntimeError("stop after first checkpoint")],
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after first checkpoint"):
                    step.run({"index": dataframe})

            manifest_after_failure = pd.read_parquet(cache_dir / "manifest.parquet")
            self.assertEqual(manifest_after_failure["image_id"].tolist(), ["img_1", "img_2"])
            self.assertTrue((cache_dir / "batches" / "batch_000000.npy").exists())

            resumed_step = ExtractEmbeddingsStep(cache_dir=cache_dir, batch_size=2)
            with patch.object(resumed_step, "_resolve_device", return_value="cpu"), patch.object(
                resumed_step, "_load_model", return_value=object()
            ), patch.object(resumed_step, "_build_transform", return_value=object()), patch.object(
                resumed_step,
                "_embed_batch",
                return_value=second_batch,
            ) as embed_batch_mock:
                resumed_step.run({"index": dataframe})

            embed_batch_mock.assert_called_once()
            manifest = pd.read_parquet(cache_dir / "manifest.parquet")
            self.assertEqual(
                manifest["image_id"].tolist(),
                ["img_1", "img_2", "img_3", "img_4"],
            )
            self.assertTrue((cache_dir / "batches" / "batch_000001.npy").exists())

    def test_progress_bar_resumes_from_cached_image_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "embedding_cache"
            (cache_dir / "batches").mkdir(parents=True)
            np.save(
                cache_dir / "batches" / "batch_000000.npy",
                np.ones((2, 4), dtype=np.float32),
            )
            pd.DataFrame(
                [
                    {"image_id": "img_1", "chunk_idx": 0, "row_idx": 0},
                    {"image_id": "img_2", "chunk_idx": 0, "row_idx": 1},
                ]
            ).to_parquet(cache_dir / "manifest.parquet", index=False)

            dataframe = pd.DataFrame(
                [
                    {"image_id": "img_1", "image_path": "/tmp/one.jpg"},
                    {"image_id": "img_2", "image_path": "/tmp/two.jpg"},
                    {"image_id": "img_3", "image_path": "/tmp/three.jpg"},
                    {"image_id": "img_4", "image_path": "/tmp/four.jpg"},
                ]
            )
            step = ExtractEmbeddingsStep(cache_dir=cache_dir, batch_size=2)
            progress = MagicMock()

            with patch.object(step, "_resolve_device", return_value="cpu"), patch.object(
                step, "_load_model", return_value=object()
            ), patch.object(step, "_build_transform", return_value=object()), patch.object(
                step,
                "_embed_batch",
                return_value=np.ones((2, step.embed_dim), dtype=np.float32),
            ), patch.object(step, "progress", return_value=nullcontext(progress)) as progress_mock:
                step.run({"index": dataframe})

            progress_mock.assert_called_once_with(
                total=4,
                initial=2,
                desc="extract embeddings",
            )
            progress.update.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
