"""Baseline models that ship with their own pretrained weights.

These models are loaded via ``torch.hub`` or from local checkpoints and
require no external training. They follow the same interface as the other
models in this package so they drop into the existing Lightning module and
configs without changes.
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from torch.nn.parameter import Parameter

try:
    from xformers.ops import memory_efficient_attention, unbind

    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False

try:
    from timm.models.layers import DropPath as _TimmDropPath
except ImportError:
    _TimmDropPath = None


class MegaLoc(nn.Module):
    """MegaLoc baseline loaded from torch.hub.

    The model ships fully pretrained — no checkpoint file is needed.

    Reference: https://github.com/gmberton/MegaLoc
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
        self.model.eval()

        # Freeze everything — this is an eval-only baseline.
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Probe the output dimensionality once so descriptor_dim is always available.
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            out = self.model(dummy)
        self.descriptor_dim: int = out.shape[-1]

    @property
    def feature_dim(self) -> int:
        return self.descriptor_dim

    def forward(self, images: Tensor) -> Tensor:
        return self.model(images)


class BoQ(nn.Module):
    """Bag-of-Queries baseline loaded from torch.hub.

    The model ships fully pretrained — no checkpoint file is needed.

    Reference: https://github.com/amaralibey/Bag-of-Queries
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.hub.load(
            "amaralibey/bag-of-queries",
            "get_trained_boq",
            backbone_name="dinov2",
            output_dim=12288,
        )
        self.model.eval()

        for param in self.model.parameters():
            param.requires_grad_(False)

        self.descriptor_dim: int = 12288

    @property
    def feature_dim(self) -> int:
        return self.descriptor_dim

    def forward(self, images: Tensor) -> Tensor:
        out = self.model(images)
        if isinstance(out, tuple):
            out = out[0]
        return out


class _HubBaseline(nn.Module):
    """Common wrapper for eval-only torch.hub baselines."""

    def __init__(self, model: nn.Module, descriptor_dim: int) -> None:
        super().__init__()
        self.model = model
        self.descriptor_dim = descriptor_dim
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return self.descriptor_dim

    def forward(self, images: Tensor) -> Tensor:
        out = self.model(images)
        if isinstance(out, tuple):
            out = out[0]
        return out


class SALAD(_HubBaseline):
    """DINOv2 SALAD baseline loaded from torch.hub.

    Reference: https://github.com/serizba/salad
    """

    def __init__(self) -> None:
        super().__init__(
            torch.hub.load("serizba/salad", "dinov2_salad"),
            descriptor_dim=8192 + 256,
        )


class EigenPlaces(_HubBaseline):
    """EigenPlaces baseline loaded from torch.hub.

    Reference: https://github.com/gmberton/EigenPlaces
    """

    def __init__(self) -> None:
        super().__init__(
            torch.hub.load(
                "gmberton/eigenplaces",
                "get_trained_model",
                backbone="ResNet50",
                fc_output_dim=2048,
            ),
            descriptor_dim=2048,
        )


class MixVPR(_HubBaseline):
    """MixVPR baseline loaded from torch.hub.

    Reference: https://github.com/jarvisyjw/MixVPR
    """

    def __init__(self) -> None:
        super().__init__(
            torch.hub.load(
                "jarvisyjw/MixVPR",
                "get_trained_model",
                pretrained=True,
            ),
            descriptor_dim=4096,
        )


class _SuperVLADLayer(nn.Module):
    """SuperVLAD aggregation layer (inference-only)."""

    def __init__(self, clusters_num: int, ghost_clusters_num: int, dim: int) -> None:
        super().__init__()
        total = clusters_num + ghost_clusters_num
        self.clusters_num = total
        self.ghost_clusters_num = ghost_clusters_num
        self.conv = nn.Conv2d(dim, total, kernel_size=(1, 1), bias=False)

    def forward(self, x: Tensor) -> Tensor:
        N, D, H, W = x.shape
        x = F.normalize(x, p=2, dim=1)
        x_flatten = x.view(N, D, -1)
        soft_assign = self.conv(x).view(N, self.clusters_num, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        vlad = torch.zeros(
            [N, self.clusters_num, D], dtype=x_flatten.dtype, device=x_flatten.device
        )
        for c in range(self.clusters_num):
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3)
            residual = residual * soft_assign[:, c : c + 1, :].unsqueeze(2)
            vlad[:, c : c + 1, :] = residual.sum(dim=-1)
        vlad = vlad[:, : -self.ghost_clusters_num, :]
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(N, -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


class SuperVLAD(nn.Module):
    """SuperVLAD baseline: DINOv2 ViT-B/14 + SuperVLAD aggregation + cross-image encoder.

    Weights are loaded from a local checkpoint shipped with the repository.
    The model is eval-only — all parameters are frozen.

    Reference: https://github.com/Lu-Feng/SuperVLAD
    """

    _CHECKPOINT = (
        Path(__file__).resolve().parents[3]
        / "checkpoints"
        / "baselines"
        / "supervlad"
        / "SuperVLAD.pth"
    )
    _BACKBONE_DIM = 768
    _CLUSTERS = 4
    _GHOST_CLUSTERS = 1

    def __init__(self) -> None:
        super().__init__()

        # ── DINOv2 ViT-B/14 backbone ─────────────────────────────────────
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

        # ── SuperVLAD aggregation ─────────────────────────────────────────
        self.aggregation = _SuperVLADLayer(
            clusters_num=self._CLUSTERS,
            ghost_clusters_num=self._GHOST_CLUSTERS,
            dim=self._BACKBONE_DIM,
        )

        # ── Cross-image encoder ──────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self._BACKBONE_DIM,
            nhead=16,
            dim_feedforward=2048,
            activation="gelu",
            dropout=0.1,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Load pretrained weights & freeze ─────────────────────────────
        self._load_checkpoint()
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

        self.descriptor_dim: int = self._CLUSTERS * self._BACKBONE_DIM  # 3072

    def _load_checkpoint(self) -> None:
        ckpt = torch.load(self._CHECKPOINT, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        # Strip ``module.`` prefix added by DDP training.
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        self.load_state_dict(sd, strict=False)

    @property
    def feature_dim(self) -> int:
        return self.descriptor_dim

    def forward(self, images: Tensor) -> Tensor:
        x = self.backbone.forward_features(images)
        B, P, D = x["x_prenorm"].shape
        W = H = int(math.sqrt(P - 1))
        x1 = x["x_norm_patchtokens"].view(B, W, H, D).permute(0, 3, 1, 2)
        x = self.aggregation(x1)
        x = self.encoder(x.view(B, -1, D)).view(B, -1)
        return F.normalize(x, p=2, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════
#  SAGE: DINOv2 ViT-L/14 + DPN recalibration + SoftP aggregation
# ═══════════════════════════════════════════════════════════════════════════


class _BackboneDPN(nn.Module):
    """Dynamic Power Normalization for backbone recalibration blocks."""

    def __init__(self, num_channels: int = 1024, eps: float = 1e-6) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(num_channels, 16),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.act = nn.Sigmoid()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x.transpose(-1, -2)).squeeze(-1)
        y = self.projection(y)
        p = self.act(y).unsqueeze(-1)
        _, L, D = x.shape
        sign = torch.sign(x)
        pw = torch.pow(torch.abs(x) + self.eps, p.expand(-1, L, D))
        return sign * pw + x


class _AggregatorDPN(nn.Module):
    """Dynamic Power Normalization for the SoftP aggregator."""

    def __init__(self, num_channels: int = 64, eps: float = 1e-6) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(num_channels, num_channels // 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(num_channels // 2, 1),
        )
        self.act = nn.Sigmoid()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x.transpose(-1, -2)).squeeze(-1)
        y = self.projection(y)
        p = self.act(y).unsqueeze(-1)
        _, L, D = x.shape
        sign = torch.sign(x)
        pw = torch.pow(torch.abs(x) + self.eps, p.expand(-1, L, D))
        return sign * pw + x


class _SoftWeighting(nn.Module):
    """Residual weighting that emphasizes informative spatial locations."""

    def __init__(self, hidden_dim: int = 128, alpha: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.predictor = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        desc = x.view(B, C, -1).transpose(1, 2)
        l2 = torch.norm(desc, p=2, dim=-1, keepdim=True) + self.eps
        beta = self.predictor(l2) * self.alpha
        beta = beta.expand(-1, -1, C)
        return (desc + beta * desc).transpose(1, 2).view(B, C, H, W)


class _DINOv2Backbone(nn.Module):
    """DINOv2 backbone with DPN recalibration on the last N blocks."""

    _ARCHS = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(
        self,
        model_name: str = "dinov2_vitl14",
        num_recalib_blocks: int = 4,
        recalibration: str = "dpn_s1",
    ) -> None:
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.num_channels = self._ARCHS[model_name]
        self.num_recalib = num_recalib_blocks
        self.recalibration = recalibration
        self.num_no_recalib = len(self.model.blocks) - self.num_recalib

        self.gpn_norms = nn.ModuleList()
        if recalibration.startswith("dpn"):
            for _ in range(num_recalib_blocks):
                self.gpn_norms.append(_BackboneDPN(num_channels=self.num_channels))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)
        for idx, blk in enumerate(self.model.blocks):
            if self.recalibration == "none" or idx < self.num_no_recalib:
                x = blk(x)
            else:
                x = x + blk.drop_path1(blk.ls1(blk.attn(blk.norm1(x))))
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
                x = self.gpn_norms[idx - self.num_no_recalib](x)
        x = self.model.norm(x)
        t = x[:, 0]
        f = x[:, 1:].reshape(B, H // 14, W // 14, self.num_channels).permute(0, 3, 1, 2)
        return f, t


class _SoftP(nn.Module):
    """SoftP aggregator producing a global place-recognition descriptor."""

    def __init__(
        self,
        num_channels: int = 1024,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim

        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.soft_weighting = _SoftWeighting(hidden_dim=128, alpha=0.5)

        self.token_features = nn.Sequential(
            nn.Linear(num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1),
            drop,
            nn.ReLU(),
            nn.Conv2d(512, cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1),
            drop,
            nn.ReLU(),
            nn.Conv2d(512, num_clusters, 1),
        )

        self.gpn = _AggregatorDPN(num_channels=num_clusters)

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        x, t = x
        x = self.soft_weighting(x)

        f = self.cluster_features(x)  # [B, cluster_dim, H, W]
        p = self.score(x)             # [B, num_clusters, H, W]

        # Remove spatial mean (before flattening)
        f = f - F.adaptive_avg_pool2d(f, (1, 1))
        p = p - F.adaptive_avg_pool2d(p, (1, 1))

        f = f.flatten(2)
        p = p.flatten(2)

        N = p.shape[2]
        p_norm = F.softmax(p, dim=2)
        fp = torch.bmm(f, p_norm.transpose(1, 2))

        # DPN post-norm
        fp_n = (1.0 / N) * fp
        fp_gpn = self.gpn(fp_n)
        fp = F.normalize(fp_gpn).flatten(1)

        t_proj = F.normalize(self.token_features(t), p=2, dim=-1)
        fp = torch.cat([t_proj, fp], dim=-1)
        return F.normalize(fp, p=2, dim=-1)


class SAGE(nn.Module):
    """SAGE baseline: DINOv2 ViT-L/14 + DPN recalibration + SoftP aggregation.

    Weights are loaded from a local checkpoint shipped with the repository.
    The model is eval-only — all parameters are frozen.

    Output descriptor dimension: 8448 (= 64 clusters x 128 dims + 256 token).
    """

    _CHECKPOINT = (
        Path(__file__).resolve().parents[3]
        / "checkpoints"
        / "baselines"
        / "sage"
        / "SAGE.pth"
    )

    def __init__(self) -> None:
        super().__init__()

        self.backbone = _DINOv2Backbone(
            model_name="dinov2_vitl14",
            num_recalib_blocks=4,
            recalibration="dpn_s1",
        )
        self.aggregator = _SoftP(
            num_channels=1024,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
        )

        self._load_checkpoint()
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

        self.descriptor_dim: int = 64 * 128 + 256  # 8448

    def _load_checkpoint(self) -> None:
        ckpt = torch.load(self._CHECKPOINT, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        self.load_state_dict(sd, strict=False)

    @property
    def feature_dim(self) -> int:
        return self.descriptor_dim

    def forward(self, images: Tensor) -> Tensor:
        x = self.backbone(images)
        return self.aggregator(x)




__all__ = ["BoQ", "EigenPlaces", "MegaLoc", "MixVPR", "SAGE", "SALAD", "SuperVLAD"]
