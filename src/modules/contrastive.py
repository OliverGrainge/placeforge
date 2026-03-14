from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from . import register_module
from .models import get_model
import pytorch_lightning as pl
from pytorch_metric_learning import losses, miners


@register_module("contrastive")
class ContrastiveLightningModule(pl.LightningModule):
    def __init__(
        self,
        *,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        val_recall_ks: list[int] | None = None,
        # MultiSimilarityLoss + MultiSimilarityMiner (designed to work together)
        ms_alpha: float = 2.0,
        ms_beta: float = 50.0,
        ms_base: float = 0.5,
        miner_epsilon: float = 0.1,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.model = get_model(model_name, **(model_kwargs or {}))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.val_recall_ks = val_recall_ks or [1, 5, 10]

        self.miner = miners.MultiSimilarityMiner(epsilon=miner_epsilon)
        self.criterion = losses.MultiSimilarityLoss(
            alpha=ms_alpha, beta=ms_beta, base=ms_base
        )

        self._val_store: dict[int, Tensor] = {}

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> Tensor:
        inputs: Tensor = batch["images"]
        labels: Tensor = batch["place_ids"]

        embeddings = self(inputs)
        hard_pairs = self.miner(embeddings, labels)
        loss: Tensor = self.criterion(embeddings, labels, hard_pairs)

        # Batch R@1: fraction of samples whose nearest neighbour shares the label
        with torch.no_grad():
            normed = torch.nn.functional.normalize(embeddings, dim=-1)
            sim = normed @ normed.T
            sim.fill_diagonal_(float("-inf"))
            nn_labels = labels[sim.argmax(dim=-1)]
            accuracy = (nn_labels == labels).float().mean()

        log_kwargs = dict(on_step=True, on_epoch=True, batch_size=labels.numel())
        self.log("train/loss", loss, prog_bar=True, **log_kwargs)
        self.log("train/accuracy", accuracy, prog_bar=False, **log_kwargs)
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_store = {}

    def validation_step(self, batch: dict[str, Any], batch_idx: int, dataloader_idx: int = 0) -> None:
        images: Tensor = batch["image"]
        image_ids: Tensor = batch["image_id"]

        with torch.no_grad():
            embeddings = self(images).cpu().float()

        if dataloader_idx not in self._val_store:
            # Pre-allocate the full embedding matrix on the first batch so that
            # any OOM failure happens immediately rather than after many batches.
            dataset = self.trainer.datamodule._val_datasets[dataloader_idx]
            embed_dim = embeddings.shape[1]
            self._val_store[dataloader_idx] = torch.empty(
                len(dataset), embed_dim, dtype=torch.float32
            )

        self._val_store[dataloader_idx][image_ids] = embeddings

    def on_validation_epoch_end(self) -> None:
        if not self._val_store:
            return

        datamodule = self.trainer.datamodule
        val_datasets = datamodule._val_datasets
        val_names = datamodule.val_dataset_names

        for dl_idx in sorted(self._val_store.keys()):
            all_embs = self._val_store[dl_idx]
            dataset = val_datasets[dl_idx]
            num_queries = dataset.num_queries

            # Records are laid out as [queries..., database...] by dataset index
            q_embs = all_embs[:num_queries].numpy()
            db_embs = all_embs[num_queries:].numpy()

            ground_truth = dataset.ground_truth()
            recalls = self._compute_recalls(q_embs, db_embs, ground_truth)

            name = val_names[dl_idx] if dl_idx < len(val_names) else str(dl_idx)
            for k, recall in recalls.items():
                self.log(f"val/{name}/R@{k}", recall, prog_bar=(k == 1))

        self._val_store.clear()

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

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


__all__ = ["ContrastiveLightningModule"]
