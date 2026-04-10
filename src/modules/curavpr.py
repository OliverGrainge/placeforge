from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .base import PlaceRecognitionModule
from .models.archs import (
    DinoV2BoQ,
    DinoV2GeM,
    DinoV2SALAD,
    SelaVPRBaseBoQ,
    SelaVPRBaseGeM,
    SelaVPRBaseSALAD,
    SelaVPRLargeBoQ,
    SelaVPRLargeGeM,
    SelaVPRLargeSALAD,
)
from pytorch_metric_learning import losses, miners

_MODELS: dict[str, type] = {
    "dinov2_gem": DinoV2GeM,
    "dinov2_salad": DinoV2SALAD,
    "dinov2_boq": DinoV2BoQ,
    "selavpr_base_boq": SelaVPRBaseBoQ,
    "selavpr_large_boq": SelaVPRLargeBoQ,
    "selavpr_base_gem": SelaVPRBaseGeM,
    "selavpr_large_gem": SelaVPRLargeGeM,
    "selavpr_base_salad": SelaVPRBaseSALAD,
    "selavpr_large_salad": SelaVPRLargeSALAD,
}

# Baseline models are registered lazily to avoid heavy torch.hub imports
# at module load time.
_BASELINE_MODELS: dict[str, tuple[str, str]] = {
    "megaloc": ("baselines", "MegaLoc"),
    "boq": ("baselines", "BoQ"),
    "salad": ("baselines", "SALAD"),
    "eigenplaces": ("baselines", "EigenPlaces"),
    "mixvpr": ("baselines", "MixVPR"),
    "supervlad": ("baselines", "SuperVLAD"),
    "sage": ("baselines", "SAGE"),
}


def _resolve_model(name: str) -> type:
    """Look up a model class by name, lazily importing baselines."""
    if name in _MODELS:
        return _MODELS[name]
    if name in _BASELINE_MODELS:
        module_name, class_name = _BASELINE_MODELS[name]
        import importlib
        mod = importlib.import_module(f".models.{module_name}", package=__package__)
        cls = getattr(mod, class_name)
        _MODELS[name] = cls  # cache for future lookups
        return cls
    return None


class CuraVPRLightningModule(PlaceRecognitionModule):
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
        lr_stages: list[dict[str, int | float]] | None = None,
        # MultiSimilarityLoss + MultiSimilarityMiner (designed to work together)
        ms_alpha: float = 1.0,
        ms_beta: float = 50.0,
        ms_base: float = 0.0,
        miner_epsilon: float = 0.1,
    ) -> None:
        super().__init__()

        if lr_schedule not in ("cosine", "constant", "staged"):
            raise ValueError(f"lr_schedule must be 'cosine', 'staged', or 'constant', got '{lr_schedule}'")

        self.save_hyperparameters()
        model_cls = _resolve_model(model_name)
        if model_cls is None:
            available = list(_MODELS) + list(_BASELINE_MODELS)
            raise ValueError(f"Unknown model {model_name!r}. Available: {available}")
        self.model = model_cls(**(model_kwargs or {}))
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.val_recall_ks = val_recall_ks or [1, 5, 10]
        self.backbone_lr_scale = backbone_lr_scale
        self.lr_schedule = lr_schedule
        self.lr_stages = lr_stages

        self.ms_miner = miners.MultiSimilarityMiner(epsilon=miner_epsilon)
        self.ms_loss = losses.MultiSimilarityLoss(
            alpha=ms_alpha, beta=ms_beta, base=ms_base
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)

    def training_step(self, batch: dict[str, Any], _batch_idx: int) -> Tensor:
        inputs: Tensor = batch["images"]
        labels: Tensor = batch["place_ids"]

        embeddings = torch.nn.functional.normalize(self(inputs), dim=-1)
        hard_pairs = self.ms_miner(embeddings, labels)
        loss: Tensor = self.ms_loss(embeddings, labels, hard_pairs)

        # Batch R@1: fraction of samples whose nearest neighbour shares the label
        with torch.no_grad():
            sim = embeddings @ embeddings.T
            sim.fill_diagonal_(float("-inf"))
            nn_labels = labels[sim.argmax(dim=-1)]
            accuracy = (nn_labels == labels).float().mean()

        log_kwargs = dict(on_step=True, on_epoch=True, batch_size=labels.numel())
        self.log("train/loss", loss, prog_bar=True, **log_kwargs)
        self.log("train/accuracy", accuracy, prog_bar=False, **log_kwargs)
        self.log("train/lr", self.optimizers().param_groups[0]["lr"], on_step=True, on_epoch=False)
        return loss

    def configure_optimizers(self):
        if hasattr(self.model, "trainable_parameters") and hasattr(self.model, "setup_for_training"):
            param_groups = [{"params": list(self.model.trainable_parameters()), "lr": self.learning_rate}]
        else:
            # Detect the backbone sub-module for differential learning rates.
            # DINOv2 models use "dino"; ResNet models use "backbone".
            backbone_module = getattr(self.model, "dino", None) or getattr(self.model, "backbone", None)
            if backbone_module is not None:
                backbone_params = list(backbone_module.parameters())
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

        if self.lr_schedule == "staged":
            stages = self.lr_stages or [{"until": 50000, "scale": 1.0}]
            cumulative = []
            step_acc = 0
            for stage in stages:
                step_acc += stage["until"]
                cumulative.append((step_acc, stage["scale"]))

            def lr_lambda(current_step: int) -> float:
                if current_step < warmup_steps:
                    return current_step / max(1, warmup_steps)
                for end_step, scale in cumulative:
                    if current_step < end_step:
                        return scale
                return cumulative[-1][1]

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

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


__all__ = ["CuraVPRLightningModule"]
