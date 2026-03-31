"""Standalone DINOv2 visual place recognition models.

Each model class is fully self-contained — it embeds its own backbone logic,
aggregation logic, and any helpers so that it can be copied into another
project without pulling in sibling classes.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

warnings.filterwarnings("ignore", message=".*xFormers is not available.*")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DinoV2GeM                                                              ║
# ║  DINOv2 ViT-B/14 + EigenPlaces-style GeM pooling with linear projection ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class DinoV2GeM(nn.Module):
    """Standalone DINOv2 + GeM pooling model.

    Backbone : DINOv2 ViT-B/14 with the last 2 blocks unfrozen.
    Aggregation: L2Norm → GeM → Linear → L2Norm (EigenPlaces-style).

    Parameters
    ----------
    descriptor_dim : int
        Output descriptor dimensionality (default 2048).
    use_checkpointing : bool
        Gradient-checkpointing for the unfrozen backbone blocks.
    """

    _BACKBONE_NAME = "dinov2_vitb14"
    _UNFREEZE_N_BLOCKS = 4
    _BACKBONE_DIM = 768

    def __init__(self, *, descriptor_dim: int = 2048, use_checkpointing: bool = True) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.use_checkpointing = use_checkpointing

        # ── backbone ──────────────────────────────────────────────────────
        self.dino = torch.hub.load("facebookresearch/dinov2", self._BACKBONE_NAME)
        for param in self.dino.parameters():
            param.requires_grad_(False)
        for block in self.dino.blocks[-self._UNFREEZE_N_BLOCKS :]:
            for param in block.parameters():
                param.requires_grad_(True)

        # ── GeM aggregation ───────────────────────────────────────────────
        self.gem_p = nn.Parameter(torch.ones(1) * 3.0)
        self.gem_eps = 1e-6
        self.projection = nn.Linear(self._BACKBONE_DIM, descriptor_dim)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output descriptor."""
        return self.descriptor_dim

    @property
    def patch_size(self) -> int:
        return self.dino.patch_embed.patch_size[0]

    # -- backbone forward --------------------------------------------------

    def _backbone_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, _, H, W = x.shape
        p = self.patch_size

        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[: -self._UNFREEZE_N_BLOCKS]:
                x = blk(x)

        for blk in self.dino.blocks[-self._UNFREEZE_N_BLOCKS :]:
            if self.use_checkpointing:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        cls_token = x[:, 0]
        spatial = x[:, 1:].permute(0, 2, 1).view(B, self._BACKBONE_DIM, H // p, W // p)
        return spatial, cls_token

    # -- aggregation forward -----------------------------------------------

    def _aggregate(self, spatial: Tensor) -> Tensor:
        x = F.normalize(spatial, p=2.0, dim=1)
        x = x.clamp(min=self.gem_eps).pow(self.gem_p).mean(dim=(-2, -1)).pow(1.0 / self.gem_p)
        return F.normalize(self.projection(x), p=2.0, dim=-1)

    # -- full forward ------------------------------------------------------

    def forward(self, images: Tensor) -> Tensor:
        spatial, _cls_token = self._backbone_forward(images)
        return self._aggregate(spatial)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DinoV2SALAD                                                            ║
# ║  DINOv2 ViT-B/14 + Sinkhorn Algorithm for Locally Aggregated Descriptors║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class DinoV2SALAD(nn.Module):
    """Standalone DINOv2 + SALAD model.

    Backbone : DINOv2 ViT-B/14 with the last 2 blocks unfrozen.
    Aggregation: SALAD (Sinkhorn Algorithm for Locally Aggregated Descriptors).

    Output size is ``num_clusters * cluster_dim + token_dim``
    (default 64 × 128 + 256 = 8 448).

    Reference: https://github.com/serizba/salad

    Parameters
    ----------
    num_clusters : int
    cluster_dim : int
    token_dim : int
    dropout : float
    use_checkpointing : bool
    """

    _BACKBONE_NAME = "dinov2_vitb14"
    _UNFREEZE_N_BLOCKS = 4
    _BACKBONE_DIM = 768

    def __init__(
        self,
        *,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.descriptor_dim = num_clusters * cluster_dim + token_dim
        self.use_checkpointing = use_checkpointing
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim

        # ── backbone ──────────────────────────────────────────────────────
        self.dino = torch.hub.load("facebookresearch/dinov2", self._BACKBONE_NAME)
        for param in self.dino.parameters():
            param.requires_grad_(False)
        for block in self.dino.blocks[-self._UNFREEZE_N_BLOCKS :]:
            for param in block.parameters():
                param.requires_grad_(True)

        # ── SALAD aggregation ─────────────────────────────────────────────
        drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_features = nn.Sequential(
            nn.Linear(self._BACKBONE_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self._BACKBONE_DIM, 512, 1),
            drop,
            nn.ReLU(),
            nn.Conv2d(512, cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(self._BACKBONE_DIM, 512, 1),
            drop,
            nn.ReLU(),
            nn.Conv2d(512, num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output descriptor."""
        return self.descriptor_dim

    @property
    def patch_size(self) -> int:
        return self.dino.patch_embed.patch_size[0]

    # -- backbone forward --------------------------------------------------

    def _backbone_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, _, H, W = x.shape
        p = self.patch_size

        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[: -self._UNFREEZE_N_BLOCKS]:
                x = blk(x)

        for blk in self.dino.blocks[-self._UNFREEZE_N_BLOCKS :]:
            if self.use_checkpointing:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        cls_token = x[:, 0]
        spatial = x[:, 1:].permute(0, 2, 1).view(B, self._BACKBONE_DIM, H // p, W // p)
        return spatial, cls_token

    # -- sinkhorn (static helper) ------------------------------------------

    @staticmethod
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

    # -- aggregation forward -----------------------------------------------

    def _aggregate(self, spatial: Tensor, cls_token: Tensor) -> Tensor:
        f = self.cluster_features(spatial).flatten(2)
        p = self.score(spatial).flatten(2)
        t = self.token_features(cls_token)

        p = torch.exp(self._sinkhorn(p, self.dust_bin))[:, :-1, :]
        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)

        descriptor = torch.cat(
            [
                F.normalize(t, p=2, dim=-1),
                F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )
        return F.normalize(descriptor, p=2, dim=-1)

    # -- full forward ------------------------------------------------------

    def forward(self, images: Tensor) -> Tensor:
        spatial, cls_token = self._backbone_forward(images)
        return self._aggregate(spatial, cls_token)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  DinoV2BoQ                                                              ║
# ║  DINOv2 ViT-B/14 + Bag-of-Queries aggregation                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


class _BoQBlock(nn.Module):
    """Single Bag-of-Queries block (internal to DinoV2BoQ)."""

    def __init__(self, in_dim: int, num_queries: int, nheads: int = 8) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=nheads,
            dim_feedforward=4 * in_dim,
            batch_first=True,
            dropout=0.0,
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


class DinoV2BoQ(nn.Module):
    """Standalone DINOv2 + Bag-of-Queries model.

    Backbone : DINOv2 ViT-B/14 with the last 2 blocks unfrozen.
    Aggregation: Bag-of-Queries.

    Output size is ``proj_channels * row_dim`` (default 384 × 32 = 12 288).

    Reference: https://github.com/amaralibey/Bag-of-Queries

    Parameters
    ----------
    proj_channels : int
    num_queries : int
    num_layers : int
    row_dim : int
    use_checkpointing : bool
    """

    _BACKBONE_NAME = "dinov2_vitb14"
    _UNFREEZE_N_BLOCKS = 4
    _BACKBONE_DIM = 768

    def __init__(
        self,
        *,
        proj_channels: int = 384,
        num_queries: int = 64,
        num_layers: int = 2,
        row_dim: int = 32,
        use_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.descriptor_dim = proj_channels * row_dim
        self.use_checkpointing = use_checkpointing

        # ── backbone ──────────────────────────────────────────────────────
        self.dino = torch.hub.load("facebookresearch/dinov2", self._BACKBONE_NAME)
        for param in self.dino.parameters():
            param.requires_grad_(False)
        for block in self.dino.blocks[-self._UNFREEZE_N_BLOCKS :]:
            for param in block.parameters():
                param.requires_grad_(True)

        # ── BoQ aggregation ───────────────────────────────────────────────
        self.proj_c = nn.Conv2d(self._BACKBONE_DIM, proj_channels, kernel_size=3, padding=1)
        self.norm_input = nn.LayerNorm(proj_channels)
        self.boqs = nn.ModuleList(
            [
                _BoQBlock(proj_channels, num_queries, nheads=proj_channels // 64)
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Linear(num_layers * num_queries, row_dim)

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the output descriptor."""
        return self.descriptor_dim

    @property
    def patch_size(self) -> int:
        return self.dino.patch_embed.patch_size[0]

    # -- backbone forward --------------------------------------------------

    def _backbone_forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        B, _, H, W = x.shape
        p = self.patch_size

        with torch.no_grad():
            x = self.dino.prepare_tokens_with_masks(x)
            for blk in self.dino.blocks[: -self._UNFREEZE_N_BLOCKS]:
                x = blk(x)

        for blk in self.dino.blocks[-self._UNFREEZE_N_BLOCKS :]:
            if self.use_checkpointing:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        cls_token = x[:, 0]
        spatial = x[:, 1:].permute(0, 2, 1).view(B, self._BACKBONE_DIM, H // p, W // p)
        return spatial, cls_token

    # -- aggregation forward -----------------------------------------------

    def _aggregate(self, spatial: Tensor) -> Tensor:
        x = self.norm_input(self.proj_c(spatial).flatten(2).permute(0, 2, 1))

        outs = []
        for block in self.boqs:
            x, out, _ = block(x)
            outs.append(out)

        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1)).flatten(1)
        return F.normalize(out, p=2, dim=-1)

    # -- full forward ------------------------------------------------------

    def forward(self, images: Tensor) -> Tensor:
        spatial, _cls_token = self._backbone_forward(images)
        return self._aggregate(spatial)


__all__ = [
    "DinoV2BoQ",
    "DinoV2GeM",
    "DinoV2SALAD",
]
