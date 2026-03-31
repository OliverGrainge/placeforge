from __future__ import annotations

import ctypes
import gc
from typing import Any

import numpy as np
import torch
from torch import Tensor

_libc = ctypes.CDLL("libc.so.6")


import pytorch_lightning as pl


class PlaceRecognitionModule(pl.LightningModule):
    """Base class providing validation and test recall@k evaluation."""

    val_recall_ks: list[int]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self) -> None:
        datamodule = self.trainer.datamodule
        self._val_store = {
            i: torch.empty(len(ds), self.model.descriptor_dim, dtype=torch.float32)
            for i, ds in enumerate(datamodule._val_datasets)
        }

    def validation_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        images: Tensor = batch["image"]
        image_ids: Tensor = batch["image_id"]

        with torch.no_grad():
            embeddings = self(images).cpu().float()

        self._val_store[dataloader_idx][image_ids] = embeddings

    def on_validation_epoch_end(self) -> None:
        if not self._val_store:
            return

        datamodule = self.trainer.datamodule
        val_datasets = datamodule._val_datasets
        val_names = datamodule.val_dataset_names

        r1_values: list[float] = []
        for dl_idx in sorted(self._val_store.keys()):
            all_embs = self._val_store.pop(dl_idx)
            dataset = val_datasets[dl_idx]
            num_queries = dataset.num_queries

            q_embs = all_embs[:num_queries].numpy().copy()
            db_embs = all_embs[num_queries:].numpy().copy()
            del all_embs

            ground_truth = dataset.ground_truth()
            recalls = self._compute_recalls(q_embs, db_embs, ground_truth)
            del q_embs, db_embs, ground_truth

            name = val_names[dl_idx] if dl_idx < len(val_names) else str(dl_idx)
            for k, recall in recalls.items():
                self.log(f"val/{name}/R@{k}", recall, prog_bar=(k == 1), add_dataloader_idx=False)
            if 1 in recalls and not np.isnan(recalls[1]):
                r1_values.append(recalls[1])

        if r1_values:
            self.log("val/R@1", float(np.mean(r1_values)))

        self._val_store.clear()
        gc.collect()
        _libc.malloc_trim(0)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def on_test_epoch_start(self) -> None:
        datamodule = self.trainer.datamodule
        self._test_store = {
            i: torch.empty(len(ds), self.model.descriptor_dim, dtype=torch.float32)
            for i, ds in enumerate(datamodule._test_datasets)
        }

    def test_step(
        self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        images: Tensor = batch["image"]
        image_ids: Tensor = batch["image_id"]

        with torch.no_grad():
            embeddings = self(images).cpu().float()

        self._test_store[dataloader_idx][image_ids] = embeddings

    def on_test_epoch_end(self) -> None:
        if not self._test_store:
            return

        datamodule = self.trainer.datamodule
        test_datasets = datamodule._test_datasets
        test_names = datamodule.test_dataset_names

        for dl_idx in sorted(self._test_store.keys()):
            all_embs = self._test_store[dl_idx]
            dataset = test_datasets[dl_idx]
            num_queries = dataset.num_queries

            q_embs = all_embs[:num_queries].numpy()
            db_embs = all_embs[num_queries:].numpy()

            ground_truth = dataset.ground_truth()
            recalls = self._compute_recalls(q_embs, db_embs, ground_truth)

            name = test_names[dl_idx] if dl_idx < len(test_names) else str(dl_idx)
            for k, recall in recalls.items():
                self.log(f"test/{name}/R@{k}", recall, prog_bar=(k == 1), add_dataloader_idx=False)

        self._test_store.clear()

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    def _compute_recalls(
        self,
        q_embs: np.ndarray,
        db_embs: np.ndarray,
        ground_truth: list[tuple[int, list[int]]],
    ) -> dict[int, float]:
        import faiss

        q_embs = q_embs.astype(np.float32)
        db_embs = db_embs.astype(np.float32)
        faiss.normalize_L2(db_embs)
        faiss.normalize_L2(q_embs)

        index = faiss.IndexFlatIP(db_embs.shape[1])
        index.add(db_embs)

        max_k = max(self.val_recall_ks)
        _, retrieved = index.search(q_embs, min(max_k, len(db_embs)))
        del index

        recalls: dict[int, float] = {}
        for k in self.val_recall_ks:
            if k > retrieved.shape[1]:
                recalls[k] = float("nan")
                continue
            correct = sum(
                bool(set(retrieved[i, :k].tolist()) & set(pos_dbids))
                for i, (_, pos_dbids) in enumerate(ground_truth)
            )
            recalls[k] = correct / len(ground_truth) if ground_truth else 0.0

        return recalls


__all__ = ["PlaceRecognitionModule"]
