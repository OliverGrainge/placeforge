from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from . import register_module
from .base import PlaceRecognitionModule
from .models import get_model
from pytorch_metric_learning import losses, miners


@register_module("contrastive")
class ContrastiveLightningModule(PlaceRecognitionModule):
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

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


__all__ = ["ContrastiveLightningModule"]
