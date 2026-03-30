from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from . import register_module
from .base import PlaceRecognitionModule
from .models import get_model


class GeneralizedContrastiveLoss(torch.nn.Module):
    """Generalized Contrastive Loss (GCL) from Leyva-Vallina et al. (2023).

    L_GCL(x_i, x_j) = ψ · ½·d² + (1 − ψ) · ½·max(τ − d, 0)²

    where d = ||f(x_i) − f(x_j)||₂,  ψ ∈ [0, 1] is the graded similarity,
    and τ is the margin separating dissimilar pairs.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        anchor_embs: Tensor,
        pair_embs: Tensor,
        similarities: Tensor,
    ) -> Tensor:
        d = torch.nn.functional.pairwise_distance(anchor_embs, pair_embs)
        pos_term = similarities * 0.5 * d.pow(2)
        neg_term = (1.0 - similarities) * 0.5 * torch.clamp(self.margin - d, min=0.0).pow(2)
        return (pos_term + neg_term).mean()


@register_module("graded_similarity")
class GradedSimilarityLightningModule(PlaceRecognitionModule):
    def __init__(
        self,
        *,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 200,
        val_recall_ks: list[int] | None = None,
        backbone_lr_scale: float = 0.1,
        lr_schedule: str = "cosine",
        margin: float = 1.0,
    ) -> None:
        super().__init__()

        if lr_schedule not in ("cosine", "constant"):
            raise ValueError(f"lr_schedule must be 'cosine' or 'constant', got '{lr_schedule}'")

        self.save_hyperparameters()
        self.model = get_model(model_name, **(model_kwargs or {}))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.val_recall_ks = val_recall_ks or [1, 5, 10]
        self.backbone_lr_scale = backbone_lr_scale
        self.lr_schedule = lr_schedule

        self.criterion = GeneralizedContrastiveLoss(margin=margin)

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> Tensor:
        anchors: Tensor = batch["anchors"]
        pairs: Tensor = batch["pairs"]
        similarities: Tensor = batch["similarities"]

        anchor_embs = self(anchors)
        pair_embs = self(pairs)

        loss: Tensor = self.criterion(anchor_embs, pair_embs, similarities)

        with torch.no_grad():
            d = torch.nn.functional.pairwise_distance(anchor_embs, pair_embs)
            pos_mask = similarities > 0.5
            neg_mask = similarities == 0.0
            mean_pos_dist = d[pos_mask].mean() if pos_mask.any() else torch.tensor(0.0)
            mean_neg_dist = d[neg_mask].mean() if neg_mask.any() else torch.tensor(0.0)

        batch_size = anchors.size(0)
        log_kwargs = dict(on_step=True, on_epoch=True, batch_size=batch_size)
        self.log("train/loss", loss, prog_bar=True, **log_kwargs)
        self.log("train/pos_dist", mean_pos_dist, **log_kwargs)
        self.log("train/neg_dist", mean_neg_dist, **log_kwargs)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        if hasattr(self.model, "backbone"):
            backbone_params = list(self.model.backbone.parameters())
            backbone_ids = {id(p) for p in backbone_params}
            head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
            param_groups = [
                {"params": backbone_params, "lr": self.learning_rate * self.backbone_lr_scale},
                {"params": head_params, "lr": self.learning_rate},
            ]
        else:
            param_groups = [{"params": list(self.parameters()), "lr": self.learning_rate}]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        warmup_steps = self.warmup_steps

        if self.lr_schedule == "constant":
            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return current_step / max(1, warmup_steps)
                return 1.0
        else:
            max_steps = self.trainer.max_steps
            min_factor = 0.01

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return current_step / max(1, warmup_steps)
                progress = (current_step - warmup_steps) / max(1, max_steps - warmup_steps)
                return min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


__all__ = ["GeneralizedContrastiveLoss", "GradedSimilarityLightningModule"]
