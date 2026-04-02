"""Lightweight ResNet-based visual place recognition models.

Intended for fast sanity-check training runs — much cheaper than DINOv2
while following the exact same interface so they drop into the existing
Lightning module and configs without any changes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  ResNet18GeM                                                            ║
# ║  ImageNet-pretrained ResNet-18 + GeM pooling with linear projection     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class ResNet18GeM(nn.Module):
    """ResNet-18 + GeM pooling model for fast sanity-check runs.

    Backbone : torchvision ResNet-18 (ImageNet-pretrained).
               Early layers (up to and including ``freeze_up_to``) are frozen;
               later layers are trainable.
    Aggregation: GeM → Linear → L2Norm (same style as DinoV2GeM).

    Parameters
    ----------
    descriptor_dim : int
        Output descriptor dimensionality (default 512).
    freeze_up_to : str
        Freeze all layers up to and including this named child of the
        ResNet.  One of ``"layer1"``, ``"layer2"``, ``"layer3"``, or
        ``"none"`` to train everything.  Default ``"layer2"``.
    """

    _BACKBONE_DIM = 512  # ResNet-18 final conv channels

    def __init__(
        self,
        *,
        descriptor_dim: int = 512,
        freeze_up_to: str = "layer2",
    ) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim

        # ── backbone ──────────────────────────────────────────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Keep everything except avgpool + fc; we replace those with GeM.
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        if freeze_up_to != "none":
            _LAYER_NAMES = {
                "layer1": 4,  # conv1 + bn1 + relu + maxpool + layer1
                "layer2": 5,
                "layer3": 6,
            }
            if freeze_up_to not in _LAYER_NAMES:
                raise ValueError(
                    f"freeze_up_to must be one of {list(_LAYER_NAMES)} or 'none', "
                    f"got {freeze_up_to!r}"
                )
            freeze_end = _LAYER_NAMES[freeze_up_to] + 1  # inclusive
            for child in list(self.backbone.children())[:freeze_end]:
                for param in child.parameters():
                    param.requires_grad_(False)

        # ── GeM aggregation ───────────────────────────────────────────────
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        self.gem_eps = 1e-6
        self.projection = nn.Linear(self._BACKBONE_DIM, descriptor_dim)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output descriptor."""
        return self.descriptor_dim

    def forward(self, images: Tensor) -> Tensor:
        spatial = self.backbone(images)  # (B, 512, H', W')
        # GeM pool → project → L2-normalise
        x = F.normalize(spatial, p=2.0, dim=1)
        x = (
            x.clamp(min=self.gem_eps)
            .pow(self.gem_p)
            .mean(dim=(-2, -1))
            .pow(1.0 / self.gem_p)
        )
        return F.normalize(self.projection(x), p=2.0, dim=-1)


__all__ = ["ResNet18GeM"]
