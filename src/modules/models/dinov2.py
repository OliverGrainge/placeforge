from __future__ import annotations

import torch
from torch import Tensor, nn

from . import register_model
import warnings

warnings.filterwarnings("ignore", message=".*xFormers is not available.*")


class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        if x.ndim == 3:
            x = x.mean(dim=1)
        elif x.ndim == 4:
            x = x.mean(dim=(-2, -1))
        else:
            raise ValueError(
                f"GeMPooling expects a 3D token tensor or 4D feature map, got shape {tuple(x.shape)}"
            )
        return x.pow(1.0 / self.p)


@register_model("dinov2_gem")
class DinoV2GeMModel(nn.Module):
    BACKBONE_NAME = "dinov2_vits14"
    PRETRAINED = True
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

        self.backbone_name = self.BACKBONE_NAME
        self.descriptor_dim = descriptor_dim
        self.backbone = self._load_backbone(
            self.BACKBONE_NAME,
            pretrained=self.PRETRAINED,
        )
        self.pool = GeMPooling(p=self.GEM_P, eps=self.EPS)

        embedding_dim = self._infer_embedding_dim()
        self.projection = nn.Linear(embedding_dim, descriptor_dim)

    def forward(self, images: Tensor) -> Tensor:
        patch_tokens = self._forward_patch_tokens(images)
        pooled = self.pool(patch_tokens)
        return self.projection(pooled)

    def _forward_patch_tokens(self, images: Tensor) -> Tensor:
        if not hasattr(self.backbone, "forward_features"):
            raise AttributeError(
                f"DINOv2 backbone {self.backbone_name!r} does not expose forward_features"
            )

        features = self.backbone.forward_features(images)

        if isinstance(features, dict):
            for key in ("x_norm_patchtokens", "x_prenorm_patchtokens"):
                value = features.get(key)
                if value is not None:
                    return value

        raise KeyError(
            "DINOv2 forward_features did not return patch tokens under "
            "'x_norm_patchtokens' or 'x_prenorm_patchtokens'"
        )

    def _infer_embedding_dim(self) -> int:
        embed_dim = getattr(self.backbone, "embed_dim", None)
        if isinstance(embed_dim, int):
            return embed_dim

        num_features = getattr(self.backbone, "num_features", None)
        if isinstance(num_features, int):
            return num_features

        raise AttributeError(
            f"Unable to infer embedding dimension from backbone {self.backbone_name!r}"
        )

    @staticmethod
    def _load_backbone(backbone_name: str, *, pretrained: bool) -> nn.Module:
        try:
            return torch.hub.load(
                "facebookresearch/dinov2",
                backbone_name,
                pretrained=pretrained,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the DINOv2 backbone via torch.hub. "
                "Ensure torch is installed and the model can be resolved."
            ) from exc


__all__ = ["DinoV2GeMModel", "GeMPooling"]
