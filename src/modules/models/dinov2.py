from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from . import register_model

warnings.filterwarnings("ignore", message=".*xFormers is not available.*")


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class DinoV2Backbone(nn.Module):
    """Pretrained DINOv2 ViT-B/14 with the last 2 blocks unfrozen.

    forward() returns (spatial [B, C, H, W], cls_token [B, C]).
    out_channels is always 768 (ViT-B embed_dim).
    """

    BACKBONE_NAME = "dinov2_vitb14"
    UNFREEZE_N_BLOCKS = 2
    out_channels = 768

    def __init__(self, use_checkpointing: bool = False) -> None:
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.dino = torch.hub.load("facebookresearch/dinov2", self.BACKBONE_NAME)

        for param in self.dino.parameters():
            param.requires_grad_(False)

        for block in self.dino.blocks[-self.UNFREEZE_N_BLOCKS:]:
            for param in block.parameters():
                param.requires_grad_(True)

    @property
    def patch_size(self) -> int:
        return self.dino.patch_embed.patch_size[0]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, _, H, W = x.shape
        p = self.patch_size

        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[: -self.UNFREEZE_N_BLOCKS]:
                x = blk(x)

        for blk in self.dino.blocks[-self.UNFREEZE_N_BLOCKS:]:
            if self.use_checkpointing:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        cls_token = x[:, 0]
        spatial = x[:, 1:].permute(0, 2, 1).view(B, self.out_channels, H // p, W // p)
        return spatial, cls_token


# ---------------------------------------------------------------------------
# Aggregation modules
# ---------------------------------------------------------------------------


class GeMPooling(nn.Module):
    """EigenPlaces-style aggregation: L2Norm → GeM → Linear → L2Norm."""

    def __init__(self, in_dim: int, descriptor_dim: int, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.projection = nn.Linear(in_dim, descriptor_dim)

    def forward(self, spatial: Tensor, cls_token: Tensor) -> Tensor:  # noqa: ARG002
        x = F.normalize(spatial, p=2.0, dim=1)
        x = x.clamp(min=self.eps).pow(self.p).mean(dim=(-2, -1)).pow(1.0 / self.p)
        return F.normalize(self.projection(x), p=2.0, dim=-1)


class BoQBlock(nn.Module):
    def __init__(self, in_dim: int, num_queries: int, nheads: int = 8) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=nheads, dim_feedforward=4 * in_dim,
            batch_first=True, dropout=0.0,
        )
        self.queries = nn.Parameter(torch.randn(1, num_queries, in_dim))
        self.self_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_q = nn.LayerNorm(in_dim)
        self.cross_attn = nn.MultiheadAttention(in_dim, num_heads=nheads, batch_first=True)
        self.norm_out = nn.LayerNorm(in_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B = x.size(0)
        x = self.encoder(x)
        q = self.queries.expand(B, -1, -1)
        q = q + self.self_attn(q, q, q)[0]
        q = self.norm_q(q)
        out, attn = self.cross_attn(q, x, x)
        return x, self.norm_out(out), attn.detach()


class BoQ(nn.Module):
    """Bag-of-Queries aggregation.

    Output shape: (B, proj_channels * row_dim), L2-normalised.
    Reference: https://github.com/amaralibey/Bag-of-Queries
    """

    def __init__(
        self,
        in_channels: int,
        proj_channels: int = 384,
        num_queries: int = 64,
        num_layers: int = 2,
        row_dim: int = 32,
    ) -> None:
        super().__init__()
        self.proj_c = nn.Conv2d(in_channels, proj_channels, kernel_size=3, padding=1)
        self.norm_input = nn.LayerNorm(proj_channels)
        self.boqs = nn.ModuleList([
            BoQBlock(proj_channels, num_queries, nheads=proj_channels // 64)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(num_layers * num_queries, row_dim)

    def forward(self, spatial: Tensor, cls_token: Tensor) -> Tensor:  # noqa: ARG002
        x = self.norm_input(self.proj_c(spatial).flatten(2).permute(0, 2, 1))

        outs = []
        for block in self.boqs:
            x, out, _ = block(x)
            outs.append(out)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1)).flatten(1)
        return F.normalize(out, p=2, dim=-1)


class SALAD(nn.Module):
    """Sinkhorn Algorithm for Locally Aggregated Descriptors.

    Output shape: (B, num_clusters * cluster_dim + token_dim), L2-normalised.
    Reference: https://github.com/serizba/salad
    """

    def __init__(
        self,
        num_channels: int,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim

        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_features = nn.Sequential(
            nn.Linear(num_channels, 512), nn.ReLU(), nn.Linear(512, token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1), drop, nn.ReLU(), nn.Conv2d(512, cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1), drop, nn.ReLU(), nn.Conv2d(512, num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, spatial: Tensor, cls_token: Tensor) -> Tensor:
        f = self.cluster_features(spatial).flatten(2)
        p = self.score(spatial).flatten(2)
        t = self.token_features(cls_token)

        p = torch.exp(_sinkhorn(p, self.dust_bin))[:, :-1, :]
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        descriptor = torch.cat([
            F.normalize(t, p=2, dim=-1),
            F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
        ], dim=-1)
        return F.normalize(descriptor, p=2, dim=-1)


class NetVLAD(nn.Module):
    """NetVLAD aggregation over a spatial feature map (B, C, H, W).

    Output shape: (B, num_clusters * in_channels), L2-normalised.
    Reference: https://github.com/Nanne/pytorch-NetVlad
    """

    def __init__(
        self,
        in_channels: int,
        num_clusters: int = 64,
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.in_channels = in_channels
        self.normalize_input = normalize_input

        self.conv = nn.Conv2d(in_channels, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(
            F.normalize(torch.randn(num_clusters, in_channels), p=2, dim=1)
        )

    def forward(self, spatial: Tensor, cls_token: Tensor) -> Tensor:  # noqa: ARG002
        B, C, H, W = spatial.shape

        if self.normalize_input:
            spatial = F.normalize(spatial, p=2, dim=1)

        # Soft assignment: (B, K, N)
        soft_assign = F.softmax(self.conv(spatial).view(B, self.num_clusters, -1), dim=1)

        # x_flat: (B, C, N)
        x_flat = spatial.view(B, C, -1)

        # VLAD: sum_x a_k(x) * (x - c_k) = sum_x a_k(x)*x - c_k * sum_x a_k(x)
        vlad = torch.einsum("bkn,bcn->bkc", soft_assign, x_flat)
        vlad -= soft_assign.sum(dim=2, keepdim=True) * self.centroids.unsqueeze(0)

        # Intra-normalisation then global L2-norm
        vlad = F.normalize(vlad, p=2, dim=2)
        return F.normalize(vlad.flatten(1), p=2, dim=1)


# ---------------------------------------------------------------------------
# Registered architectures
# ---------------------------------------------------------------------------


@register_model("dinov2_gem")
class DinoV2GeM(nn.Module):
    """DINOv2 + EigenPlaces GeM pooling with linear projection."""

    def __init__(self, *, descriptor_dim: int = 2048, use_checkpointing: bool = False) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.backbone = DinoV2Backbone(use_checkpointing=use_checkpointing)
        self.aggregation = GeMPooling(DinoV2Backbone.out_channels, descriptor_dim)

    def forward(self, images: Tensor) -> Tensor:
        spatial, cls_token = self.backbone(images)
        return self.aggregation(spatial, cls_token)


@register_model("dinov2_salad")
class DinoV2SALAD(nn.Module):
    """DINOv2 + SALAD aggregation.

    Output size is num_clusters * cluster_dim + token_dim (default 64*128+256 = 8448).
    """

    def __init__(
        self,
        *,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.descriptor_dim = num_clusters * cluster_dim + token_dim
        self.backbone = DinoV2Backbone(use_checkpointing=use_checkpointing)
        self.aggregation = SALAD(
            num_channels=DinoV2Backbone.out_channels,
            num_clusters=num_clusters,
            cluster_dim=cluster_dim,
            token_dim=token_dim,
            dropout=dropout,
        )

    def forward(self, images: Tensor) -> Tensor:
        spatial, cls_token = self.backbone(images)
        return self.aggregation(spatial, cls_token)


@register_model("dinov2_netvlad")
class DinoV2NetVLAD(nn.Module):
    """DINOv2 + NetVLAD aggregation.

    Output size is num_clusters * 768 (default 64 * 768 = 49152).
    """

    def __init__(
        self,
        *,
        num_clusters: int = 64,
        normalize_input: bool = True,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.descriptor_dim = num_clusters * DinoV2Backbone.out_channels
        self.backbone = DinoV2Backbone(use_checkpointing=use_checkpointing)
        self.aggregation = NetVLAD(
            in_channels=DinoV2Backbone.out_channels,
            num_clusters=num_clusters,
            normalize_input=normalize_input,
        )

    def forward(self, images: Tensor) -> Tensor:
        spatial, cls_token = self.backbone(images)
        return self.aggregation(spatial, cls_token)


@register_model("dinov2_boq")
class DinoV2BoQ(nn.Module):
    """DINOv2 + Bag-of-Queries aggregation.

    Output size is proj_channels * row_dim (default 384 * 32 = 12288).
    """

    def __init__(
        self,
        *,
        proj_channels: int = 384,
        num_queries: int = 64,
        num_layers: int = 2,
        row_dim: int = 32,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.descriptor_dim = proj_channels * row_dim
        self.backbone = DinoV2Backbone(use_checkpointing=use_checkpointing)
        self.aggregation = BoQ(
            in_channels=DinoV2Backbone.out_channels,
            proj_channels=proj_channels,
            num_queries=num_queries,
            num_layers=num_layers,
            row_dim=row_dim,
        )

    def forward(self, images: Tensor) -> Tensor:
        spatial, cls_token = self.backbone(images)
        return self.aggregation(spatial, cls_token)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sinkhorn(S: Tensor, dustbin_score: Tensor, num_iters: int = 3, reg: float = 1.0) -> Tensor:
    B, m, n = S.shape
    S_aug = torch.cat([S, dustbin_score.expand(B, 1, n)], dim=1)

    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a = norm.expand(B, m + 1).clone()
    log_a[:, -1] += math.log(n - m)
    log_b = norm.expand(B, n)

    M = S_aug / reg
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1)

    return M + u.unsqueeze(2) + v.unsqueeze(1) - norm


__all__ = [
    "BoQ",
    "DinoV2Backbone",
    "DinoV2BoQ",
    "DinoV2GeM",
    "DinoV2NetVLAD",
    "DinoV2SALAD",
    "GeMPooling",
    "NetVLAD",
    "SALAD",
]
