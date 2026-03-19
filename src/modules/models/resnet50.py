from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import ResNet50_Weights, resnet50

from . import register_model


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class ResNet50Backbone(nn.Module):
    """Pretrained ResNet-50 with the stem and layer1 frozen.

    out_channels is always 2048.
    """

    out_channels = 2048

    def __init__(self) -> None:
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        self.conv1   = m.conv1
        self.bn1     = m.bn1
        self.relu    = m.relu
        self.maxpool = m.maxpool
        self.layer1  = m.layer1
        self.layer2  = m.layer2
        self.layer3  = m.layer3
        self.layer4  = m.layer4

        # freeze stem and layer1
        for p in [*self.conv1.parameters(), *self.bn1.parameters(), *self.layer1.parameters()]:
            p.requires_grad_(False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# ---------------------------------------------------------------------------
# Aggregation modules
# ---------------------------------------------------------------------------


class SpatialGeMPooling(nn.Module):
    """Generalised Mean Pooling over a spatial feature map."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, features: Tensor) -> Tensor:
        x = features.clamp(min=self.eps).pow(self.p)
        return x.mean(dim=(-2, -1)).pow(1.0 / self.p)


class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim: int, mlp_ratio: float = 1.0) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mix(x)


class MixVPR(nn.Module):
    """MixVPR aggregation over a spatial feature map (B, C, H, W).

    Output shape: (B, out_channels * out_rows), L2-normalised.

    Reference: https://github.com/amaralibey/MixVPR
    """

    def __init__(
        self,
        in_channels: int,
        in_h: int,
        in_w: int,
        out_channels: int = 512,
        mix_depth: int = 1,
        mlp_ratio: float = 1.0,
        out_rows: int = 4,
    ) -> None:
        super().__init__()
        hw = in_h * in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio)
            for _ in range(mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(2)
        x = self.mix(x)
        x = self.channel_proj(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.row_proj(x)
        return F.normalize(x.flatten(1), p=2, dim=-1)


# ---------------------------------------------------------------------------
# Registered architectures
# ---------------------------------------------------------------------------


@register_model("resnet50_gem")
class ResNet50GeM(nn.Module):
    """ResNet-50 + Generalised Mean Pooling + linear projection."""

    def __init__(self, *, descriptor_dim: int) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.backbone = ResNet50Backbone()
        self.aggregation = SpatialGeMPooling()
        self.projection = nn.Linear(self.backbone.out_channels, descriptor_dim)

    def forward(self, images: Tensor) -> Tensor:
        return self.projection(self.aggregation(self.backbone(images)))


@register_model("resnet50_mixvpr")
class ResNet50MixVPR(nn.Module):
    """ResNet-50 + MixVPR aggregation.

    Output size is out_channels * out_rows (default 512 * 4 = 2048).
    in_h / in_w are the spatial dims of the backbone feature map —
    7x7 for 224x224 input with no layer cropping.
    """

    def __init__(
        self,
        *,
        out_channels: int = 512,
        mix_depth: int = 1,
        mlp_ratio: float = 1.0,
        out_rows: int = 4,
        in_h: int = 7,
        in_w: int = 7,
    ) -> None:
        super().__init__()
        self.descriptor_dim = out_channels * out_rows
        self.backbone = ResNet50Backbone()
        self.aggregation = MixVPR(
            in_channels=self.backbone.out_channels,
            in_h=in_h,
            in_w=in_w,
            out_channels=out_channels,
            mix_depth=mix_depth,
            mlp_ratio=mlp_ratio,
            out_rows=out_rows,
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.aggregation(self.backbone(images))


__all__ = [
    "FeatureMixerLayer",
    "MixVPR",
    "ResNet50Backbone",
    "ResNet50GeM",
    "ResNet50MixVPR",
    "SpatialGeMPooling",
]
