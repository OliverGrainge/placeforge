from __future__ import annotations

import itertools
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
        self.val_recall_ks = val_recall_ks or [1, 5, 10]
        self._scale = scale
        self._margin = margin

        self._supergroup_to_criterion_idx: dict[int, int] = {}
        self.criterions = torch.nn.ModuleList()
        self._active_criterion_idx: int | None = None

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

    def on_train_start(self) -> None:
        for criterion in self.criterions:
            criterion.cpu()

    def _swap_criterion(self, idx: int) -> None:
        if idx == self._active_criterion_idx:
            return
        if self._active_criterion_idx is not None:
            self._move_criterion(self._active_criterion_idx, torch.device("cpu"))
        self._move_criterion(idx, self.device)
        self._active_criterion_idx = idx

    def _move_criterion(self, idx: int, device: torch.device) -> None:
        self.criterions[idx].to(device)
        optimizer = self.optimizers()
        if optimizer is None:
            return
        for param in self.criterions[idx].parameters():
            if param in optimizer.state:
                for key, val in optimizer.state[param].items():
                    if isinstance(val, torch.Tensor):
                        optimizer.state[param][key] = val.to(device)

    def on_train_batch_start(self, batch: dict[str, Any], batch_idx: int) -> None:
        supergroup_id: int = batch["supergroup_id"].item()
        self._swap_criterion(self._supergroup_to_criterion_idx[supergroup_id])

    def on_train_epoch_end(self) -> None:
        if self._active_criterion_idx is not None:
            self._move_criterion(self._active_criterion_idx, torch.device("cpu"))
            self._active_criterion_idx = None

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
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        criterion_params = list(itertools.chain.from_iterable(
            c.parameters() for c in self.criterions
        ))
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
                {"params": criterion_params, "lr": self.learning_rate, "weight_decay": 0.0},
            ]
        else:
            non_criterion_params = [p for p in self.parameters() if id(p) not in criterion_ids]
            param_groups = [
                {"params": non_criterion_params, "lr": self.learning_rate},
                {"params": criterion_params, "lr": self.learning_rate, "weight_decay": 0.0},
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
