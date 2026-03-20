from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from . import register_module
from .base import PlaceRecognitionModule
from .models import get_model
from pytorch_metric_learning import losses


@register_module("classification")
class ClassificationLightningModule(PlaceRecognitionModule):
    def __init__(
        self,
        *,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        val_recall_ks: list[int] | None = None,
        # Large Margin Cosine Loss (CosFace) hyperparameters
        scale: float = 64.0,
        margin: float = 0.35,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.model = get_model(model_name, **(model_kwargs or {}))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.val_recall_ks = val_recall_ks or [1, 5, 10]
        self._scale = scale
        self._margin = margin

        self._supergroup_to_criterion_idx: dict[int, int] = {}
        self.criterions = torch.nn.ModuleList()

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit") and not self.criterions:
            supergroup_num_places: dict[int, int] = self.trainer.datamodule.supergroup_num_places
            sorted_supergroups = sorted(supergroup_num_places.keys())
            self._supergroup_to_criterion_idx = {sg: i for i, sg in enumerate(sorted_supergroups)}
            self.criterions = torch.nn.ModuleList([
                losses.CosFaceLoss(
                    num_classes=supergroup_num_places[sg],
                    embedding_size=self.model.descriptor_dim,
                    scale=self._scale,
                    margin=self._margin,
                )
                for sg in sorted_supergroups
            ])

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> Tensor:
        inputs: Tensor = batch["images"]
        labels: Tensor = batch["place_ids"]
        supergroup_id: int = batch["supergroup_id"].item()

        criterion = self.criterions[self._supergroup_to_criterion_idx[supergroup_id]]

        embeddings = self(inputs)
        loss: Tensor = criterion(embeddings, labels)

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
            self.parameters(),  # includes all criterions via ModuleList
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


__all__ = ["ClassificationLightningModule"]
