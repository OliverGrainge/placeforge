from __future__ import annotations

import math
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
        warmup_steps: int = 200,
        backbone_lr_scale: float = 0.1,
        criterion_lr_scale: float = 10.0,
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
        self.warmup_steps = warmup_steps
        self.backbone_lr_scale = backbone_lr_scale
        self.criterion_lr_scale = criterion_lr_scale
        self.val_recall_ks = val_recall_ks or [1, 5, 10]
        self._scale = scale
        self._margin = margin

        self.criterion: torch.nn.Module | None = None

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit") and self.criterion is None:
            num_places: int = self.trainer.datamodule.num_places
            self.criterion = losses.CosFaceLoss(
                num_classes=num_places,
                embedding_size=self.model.descriptor_dim,
                scale=self._scale,
                margin=self._margin,
            )

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> Tensor:
        inputs: Tensor = batch["images"]
        labels: Tensor = batch["labels"]

        embeddings = self(inputs)
        loss: Tensor = self.criterion(embeddings, labels)

        log_kwargs = dict(on_step=True, on_epoch=True, batch_size=labels.numel())
        self.log("train/loss", loss, prog_bar=True, **log_kwargs)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        criterion_params = list(self.criterion.parameters())
        criterion_ids = {id(p) for p in criterion_params}

        if hasattr(self.model, "backbone"):
            backbone_params = list(self.model.backbone.parameters())
            backbone_ids = {id(p) for p in backbone_params}
            head_params = [
                p for p in self.parameters()
                if id(p) not in backbone_ids and id(p) not in criterion_ids
            ]
            param_groups = [
                {"params": backbone_params, "lr": self.learning_rate * self.backbone_lr_scale},
                {"params": head_params, "lr": self.learning_rate},
                {"params": criterion_params, "lr": self.learning_rate * self.criterion_lr_scale, "weight_decay": 0.0},
            ]
        else:
            non_criterion_params = [p for p in self.parameters() if id(p) not in criterion_ids]
            param_groups = [
                {"params": non_criterion_params, "lr": self.learning_rate},
                {"params": criterion_params, "lr": self.learning_rate * self.criterion_lr_scale, "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        max_steps = self.trainer.max_steps
        warmup_steps = self.warmup_steps
        min_factor = 0.01

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
            return min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


__all__ = ["ClassificationLightningModule"]
