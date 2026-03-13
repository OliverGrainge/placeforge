from __future__ import annotations

from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from . import register_model
from .dinov2 import GeMPooling


@register_model("resnet50_gem")
class ResNet50GeMModel(nn.Module):
    PRETRAINED = False
    GEM_P = 3.0
    EPS = 1e-6

    def __init__(
        self,
        *,
        descriptor_dim: int,
    ) -> None:
        super().__init__()

        if descriptor_dim <= 0:
            raise ValueError("descriptor_dim must be positive")

        self.descriptor_dim = descriptor_dim
        self.backbone = self._load_backbone(pretrained=self.PRETRAINED)
        self.pool = GeMPooling(p=self.GEM_P, eps=self.EPS)
        self.projection = nn.Linear(2048, descriptor_dim)

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone(images)
        pooled = self.pool(features)
        return self.projection(pooled)

    @staticmethod
    def _load_backbone(*, pretrained: bool) -> nn.Module:
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        return nn.Sequential(*list(backbone.children())[:-2])


__all__ = ["ResNet50GeMModel"]
