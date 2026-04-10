"""Visual place recognition model architectures.

This module contains:

1. DINOv2-based models (GeM, SALAD, BoQ aggregation heads).
2. SelaVPR++ memory-efficient adaptation architecture with side adapter networks.

Each DINOv2 model class is fully self-contained — it embeds its own backbone
logic, aggregation logic, and any helpers so that it can be copied into another
project without pulling in sibling classes.

The SelaVPR++ models use a frozen DINOv2 backbone with a parallel side adapter
network of MultiConv adapters that progressively refines intermediate backbone
features *without* backpropagating through the frozen backbone.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Iterable, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
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

    def __init__(self, *, descriptor_dim: int = 2048, use_checkpointing: bool = True, unfreeze_n_blocks: int = 4) -> None:
        super().__init__()
        self.descriptor_dim = descriptor_dim
        self.use_checkpointing = use_checkpointing
        self._unfreeze_n_blocks = unfreeze_n_blocks

        # ── backbone ──────────────────────────────────────────────────────
        self.dino = torch.hub.load("facebookresearch/dinov2", self._BACKBONE_NAME)
        for param in self.dino.parameters():
            param.requires_grad_(False)
        for block in self.dino.blocks[-self._unfreeze_n_blocks :]:
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
            for blk in self.dino.blocks[: -self._unfreeze_n_blocks]:
                x = blk(x)

        for blk in self.dino.blocks[-self._unfreeze_n_blocks :]:
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
        unfreeze_n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.descriptor_dim = num_clusters * cluster_dim + token_dim
        self.use_checkpointing = use_checkpointing
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self._unfreeze_n_blocks = unfreeze_n_blocks

        # ── backbone ──────────────────────────────────────────────────────
        self.dino = torch.hub.load("facebookresearch/dinov2", self._BACKBONE_NAME)
        for param in self.dino.parameters():
            param.requires_grad_(False)
        for block in self.dino.blocks[-self._unfreeze_n_blocks :]:
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
            for blk in self.dino.blocks[: -self._unfreeze_n_blocks]:
                x = blk(x)

        for blk in self.dino.blocks[-self._unfreeze_n_blocks :]:
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
        unfreeze_n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.descriptor_dim = proj_channels * row_dim
        self.use_checkpointing = use_checkpointing
        self._unfreeze_n_blocks = unfreeze_n_blocks

        # ── backbone ──────────────────────────────────────────────────────
        self.dino = torch.hub.load("facebookresearch/dinov2", self._BACKBONE_NAME)
        for param in self.dino.parameters():
            param.requires_grad_(False)
        for block in self.dino.blocks[-self._unfreeze_n_blocks :]:
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
            for blk in self.dino.blocks[: -self._unfreeze_n_blocks]:
                x = blk(x)

        for blk in self.dino.blocks[-self._unfreeze_n_blocks :]:
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


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SelaVPR++ models                                                       ║
# ║  DINOv2 backbone + side adapter network + various aggregation heads     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


BackboneName = Literal["dinov2-base", "dinov2-large"]
AggregationName = Literal["gem", "boq", "salad"]
TrainMode = Literal["standard", "hashing", "rerank"]

_DINOV2_PRETRAIN_URLS: dict[BackboneName, str] = {
    "dinov2-base": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "dinov2-large": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
}


@dataclass
class SelaVPRConfig:
    backbone: BackboneName = "dinov2-large"
    aggregation: AggregationName = "gem"
    hashing: bool = False
    rerank: bool = False
    foundation_model_path: Optional[str] = None
    load_pretrained_backbone: bool = True
    resume: bool = False


def _make_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


# ---------------------------------------------------------------------------
# ViT building blocks (frozen backbone)
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        image_hw = _make_2tuple(img_size)
        patch_hw = _make_2tuple(patch_size)
        self.img_size = image_hw
        self.patch_size = patch_hw
        self.num_patches = (image_hw[0] // patch_hw[0]) * (image_hw[1] // patch_hw[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        patch_h, patch_w = self.patch_size
        assert height % patch_h == 0, f"Input height {height} is not divisible by patch height {patch_h}"
        assert width % patch_w == 0, f"Input width {width} is not divisible by patch width {patch_w}"
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    """Standard ViT block (no adapters -- adapters live in the side network)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, proj_bias, attn_drop, drop)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop, bias=ffn_bias)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Frozen DINOv2 backbone
# ---------------------------------------------------------------------------


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 518,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        init_values: Optional[float] = 1.0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        block_chunks: int = 0,
    ) -> None:
        super().__init__()
        if block_chunks != 0:
            raise ValueError("This single-file wrapper only supports block_chunks=0")

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_tokens = 1
        self.num_register_tokens = 0
        self.patch_size = patch_size
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.init_weights()

    def init_weights(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def interpolate_pos_encoding(self, x: Tensor, height: int, width: int) -> Tensor:
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        n_pos = self.pos_embed.shape[1] - 1
        if npatch == n_pos and height == width:
            return self.pos_embed

        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        height0 = height // self.patch_size
        width0 = width // self.patch_size
        pos_size = int(math.sqrt(n_pos))
        assert n_pos == pos_size * pos_size

        if self.interpolate_offset:
            kwargs = {
                "scale_factor": (
                    float(height0 + self.interpolate_offset) / pos_size,
                    float(width0 + self.interpolate_offset) / pos_size,
                )
            }
        else:
            kwargs = {"size": (height0, width0)}

        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, pos_size, pos_size, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        assert (height0, width0) == patch_pos_embed.shape[-2:]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens(self, x: Tensor, masks: Optional[Tensor] = None) -> Tensor:
        batch, _, height, width = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        return x + self.interpolate_pos_encoding(x, height, width)

    def forward_features(self, x: Tensor, masks: Optional[Tensor] = None) -> dict[str, Tensor | list[Tensor] | None]:
        """Run the frozen backbone, returning intermediate block outputs.

        Returns a dict with:
        - ``block_outputs``: list of length ``depth + 1``.  Index 0 is the
          prepared token sequence (before any block); indices 1..depth are
          the outputs of each transformer block.
        - ``x_norm_clstoken``, ``x_norm_patchtokens``, ``x_prenorm``: the
          final backbone representations (after the last block + LayerNorm).
        """
        x = self.prepare_tokens(x, masks)
        block_outputs: list[Tensor] = [x]
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)
        x_norm = self.norm(x)
        return {
            "block_outputs": block_outputs,
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x: Tensor, masks: Optional[Tensor] = None) -> dict[str, Tensor | list[Tensor] | None]:
        return self.forward_features(x, masks)


def vit_base(
    patch_size: int = 14,
    img_size: int = 518,
    **kwargs,
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        img_size=img_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        **kwargs,
    )


def vit_large(
    patch_size: int = 14,
    img_size: int = 518,
    **kwargs,
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        img_size=img_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# MultiConv adapter and side adapter network (SelaVPR++ Eq. 6)
# ---------------------------------------------------------------------------


class _BasicConv2d(nn.Module):
    """Conv2d + BatchNorm2d + ReLU, matching the reference SelaVPR++ implementation."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


class MultiConvAdapter(nn.Module):
    """Multi-scale convolution adapter from the SelaVPR++ paper.

    Three parallel conv paths (1x1, 3x3, 5x5) with BatchNorm between a
    down-projection and an up-projection.  Operates on patch tokens reshaped
    to a spatial grid; the CLS token is projected through D_fc1 and carried
    along so the internal skip connection preserves it.
    """

    def __init__(self, features: int, bottleneck_ratio: float = 0.5, skip_connect: bool = False) -> None:
        super().__init__()
        self.skip_connect = skip_connect
        hidden_features = max(1, int(features * bottleneck_ratio))
        # Channel reduction dim before 3x3 and 5x5 convs
        reduce_dim = max(1, int(24 * features / 768))

        self.D_fc1 = nn.Linear(features, hidden_features)

        # --- three parallel conv paths (inception-style with BN+ReLU) ---
        out_a = hidden_features // 2  # 192 for base, 256 for large
        self.branch1 = _BasicConv2d(hidden_features, out_a, kernel_size=1)

        out_b = hidden_features // 4  # 96 for base, 128 for large
        self.branch2 = nn.Sequential(
            _BasicConv2d(hidden_features, reduce_dim, kernel_size=1),
            _BasicConv2d(reduce_dim, out_b, kernel_size=3, padding=1),
        )

        out_c = hidden_features - out_a - out_b  # 96 for base, 128 for large
        self.branch3 = nn.Sequential(
            _BasicConv2d(hidden_features, reduce_dim, kernel_size=1),
            _BasicConv2d(reduce_dim, out_c, kernel_size=5, padding=2),
        )

        self.D_fc2 = nn.Linear(hidden_features, features)

    def forward(self, x: Tensor) -> Tensor:
        x0 = F.relu(self.D_fc1(x), inplace=False)
        batch, tokens, dim = x0.shape
        height = width = int(math.sqrt(tokens - 1))

        # Patch tokens -> spatial grid for convolutions
        xs = x0[:, 1:, :]
        xs = xs.reshape(batch, height, width, dim).permute(0, 3, 1, 2)
        b1 = self.branch1(xs)
        b2 = self.branch2(xs)
        b3 = self.branch3(xs)
        outputs = torch.cat([b1, b2, b3], dim=1)
        outputs = outputs.reshape(batch, dim, height * width).permute(0, 2, 1)

        # Recombine with CLS token from projected features
        cls_token = x0[:, :1, :]
        outputs = torch.cat([cls_token, outputs], dim=1)

        # Internal skip connection (add projected input back before D_fc2)
        outputs = outputs + x0
        outputs = self.D_fc2(outputs)

        if self.skip_connect:
            outputs = outputs + x
        return outputs


class SideAdapterNetwork(nn.Module):
    """Parallel side network that progressively refines frozen backbone outputs.

    Implements Eq. 6 from the SelaVPR++ paper::

        y_1 = Adapter(x_0 + x_1) + x_0
        y_l = Adapter(y_{l-1} + x_l) + y_{l-1}   for l >= 2

    where ``x_0, x_1, ..., x_N`` are the frozen backbone intermediate outputs
    and ``y_N`` is the final adapted representation.
    """

    def __init__(
        self,
        features: int,
        num_adapters: int,
        bottleneck_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [MultiConvAdapter(features, bottleneck_ratio) for _ in range(num_adapters)]
        )
        self.norm = nn.LayerNorm(features, eps=1e-6)

    def forward(self, block_outputs: list[Tensor]) -> Tensor:
        """
        Args:
            block_outputs: list of length ``num_adapters + 1``.
                Index 0 is the input to the first adapted block (x_0).
                Indices 1..N are the outputs of the N adapted backbone blocks.
        """
        y: Tensor = None  # type: ignore[assignment]
        for l, adapter in enumerate(self.adapters):
            x_prev = block_outputs[l]
            x_curr = block_outputs[l + 1]
            if l == 0:
                y = adapter(x_prev + x_curr) + x_prev
            else:
                y = adapter(y + x_curr) + y
        return self.norm(y)


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------


class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


# ---------------------------------------------------------------------------
# Aggregation heads
# ---------------------------------------------------------------------------


class L2Norm(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=2, dim=self.dim)


class Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[2] == x.shape[3] == 1
        return x[:, :, 0, 0]


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)


class BoQBlock(nn.Module):
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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.encoder(x)
        q = self.queries.repeat(x.size(0), 1, 1)
        q = self.norm_q(q + self.self_attn(q, q, q)[0])
        out = self.norm_out(self.cross_attn(q, x, x)[0])
        return x, out


class BoQ(nn.Module):
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
        self.boqs = nn.ModuleList(
            [BoQBlock(proj_channels, num_queries, nheads=proj_channels // 64) for _ in range(num_layers)]
        )
        self.fc = nn.Linear(num_layers * num_queries, row_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj_c(x).flatten(2).permute(0, 2, 1)
        x = self.norm_input(x)
        outs = []
        for boq in self.boqs:
            x, out = boq(x)
            outs.append(out)
        out = torch.cat(outs, dim=1)
        out = self.fc(out.permute(0, 2, 1)).flatten(1)
        return F.normalize(out, p=2, dim=-1)


def log_otp_solver(log_a: Tensor, log_b: Tensor, scores: Tensor, num_iters: int = 20, reg: float = 1.0) -> Tensor:
    scores = scores / reg
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(num_iters):
        u = log_a - torch.logsumexp(scores + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(scores + u.unsqueeze(2), dim=1).squeeze()
    return scores + u.unsqueeze(2) + v.unsqueeze(1)


def get_matching_probs(scores: Tensor, dustbin_score: Tensor, num_iters: int = 3, reg: float = 1.0) -> Tensor:
    batch_size, m, n = scores.size()
    scores_aug = torch.empty(batch_size, m + 1, n, dtype=scores.dtype, device=scores.device)
    scores_aug[:, :m, :n] = scores
    scores_aug[:, m, :] = dustbin_score

    norm = -torch.tensor(math.log(n + m), device=scores.device)
    log_a = norm.expand(m + 1).contiguous()
    log_b = norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a = log_a.expand(batch_size, -1)
    log_b = log_b.expand(batch_size, -1)
    return log_otp_solver(log_a, log_b, scores_aug, num_iters=num_iters, reg=reg) - norm


class SALAD(nn.Module):
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
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.token_features = nn.Sequential(nn.Linear(num_channels, 512), nn.ReLU(), nn.Linear(512, token_dim))
        self.cluster_features = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, num_clusters, 1),
        )
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: tuple[Tensor, Tensor]) -> Tensor:
        patch_tokens, cls_token = x
        f = self.cluster_features(patch_tokens).flatten(2)
        p = self.score(patch_tokens).flatten(2)
        t = self.token_features(cls_token)

        p = torch.exp(get_matching_probs(p, self.dust_bin, num_iters=3))
        p = p[:, :-1, :]
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


# ---------------------------------------------------------------------------
# SelaVPR++ trainable model
# ---------------------------------------------------------------------------


# Per-backbone adapter configuration: (start_block, num_adapters)
_ADAPTER_CFG: dict[BackboneName, tuple[int, int]] = {
    "dinov2-base": (0, 12),   # all 12 blocks
    "dinov2-large": (8, 16),  # last 16 of 24 blocks
}


class SelaVPRTrainable(nn.Module):
    def __init__(
        self,
        config: Optional[SelaVPRConfig] = None,
        *,
        backbone: BackboneName = "dinov2-large",
        aggregation: AggregationName = "gem",
        hashing: bool = False,
        rerank: bool = False,
        foundation_model_path: Optional[str] = None,
        pretrained_url: Optional[str] = None,
        load_pretrained_backbone: bool = True,
        resume: bool = False,
        setup_training: bool = False,
    ) -> None:
        super().__init__()
        self.config = config or SelaVPRConfig(
            backbone=backbone,
            aggregation=aggregation,
            hashing=hashing,
            rerank=rerank,
            foundation_model_path=foundation_model_path,
            load_pretrained_backbone=load_pretrained_backbone,
            resume=resume,
        )
        self.pretrained_url = pretrained_url
        self.hashing = self.config.hashing
        self.rerank = self.config.rerank
        self.aggregation_name = self.config.aggregation
        self.backbone = self._make_backbone()

        input_dim = self.input_dim
        output_dim = 2048 if input_dim == 768 else 4096
        self.descriptor_dim = self.float_descriptor_dim

        # --- Side adapter network(s) ---
        start_block, num_adapters = _ADAPTER_CFG[self.config.backbone]
        self._adapter_start = start_block

        if not self.hashing or self.rerank:
            self.side_adapter = SideAdapterNetwork(input_dim, num_adapters)
            self.aggregation = self._make_float_aggregation(input_dim)
            if self.aggregation_name == "gem":
                self.linear1 = nn.Linear(input_dim, input_dim)
                self.linear2 = nn.Linear(input_dim, output_dim)

        if self.hashing:
            self.side_adapter_hashing = SideAdapterNetwork(input_dim, num_adapters)
            self.aggregation_hashing = nn.Sequential(L2Norm(), GeM(), Flatten())
            self.linear3 = nn.Linear(input_dim, input_dim)
            self.linear4 = nn.Linear(input_dim, 512)

        if setup_training:
            self.setup_for_training()

    @property
    def feature_dim(self) -> int:
        return self.descriptor_dim

    @property
    def input_dim(self) -> int:
        if self.config.backbone == "dinov2-base":
            return 768
        if self.config.backbone == "dinov2-large":
            return 1024
        raise ValueError(f"Unknown backbone: {self.config.backbone}")

    @property
    def float_descriptor_dim(self) -> int:
        if self.aggregation_name == "gem":
            return 2048 if self.input_dim == 768 else 4096
        if self.aggregation_name == "boq":
            return 12288
        if self.aggregation_name == "salad":
            return 8448
        raise ValueError(f"Unknown aggregation: {self.aggregation_name}")

    @property
    def binary_descriptor_dim(self) -> int:
        return 512

    def _make_backbone(self) -> nn.Module:
        if self.config.backbone == "dinov2-base":
            backbone = vit_base(patch_size=14, img_size=518, init_values=1, block_chunks=0)
        elif self.config.backbone == "dinov2-large":
            backbone = vit_large(patch_size=14, img_size=518, init_values=1, block_chunks=0)
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")

        if not self.config.resume and (self.config.foundation_model_path or self.config.load_pretrained_backbone):
            state = self._load_backbone_state()
            state = {self._remove_prefix(self._remove_prefix(k, "module."), "backbone."): v for k, v in state.items()}
            current = backbone.state_dict()
            compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
            current.update(compatible)
            backbone.load_state_dict(current)
        return backbone

    def _load_backbone_state(self) -> dict[str, Tensor]:
        if self.config.foundation_model_path:
            state = torch.load(self.config.foundation_model_path, map_location="cpu")
        else:
            state = torch.hub.load_state_dict_from_url(
                self.pretrained_url or _DINOV2_PRETRAIN_URLS[self.config.backbone],
                map_location="cpu",
                progress=True,
            )
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return state

    @staticmethod
    def _remove_prefix(text: str, prefix: str) -> str:
        return text[len(prefix):] if text.startswith(prefix) else text

    def _make_float_aggregation(self, input_dim: int) -> nn.Module:
        if self.aggregation_name == "gem":
            return nn.Sequential(L2Norm(), GeM(), Flatten())
        if self.aggregation_name == "boq":
            return BoQ(in_channels=input_dim, proj_channels=384, num_queries=64, num_layers=2, row_dim=32)
        if self.aggregation_name == "salad":
            return SALAD(num_channels=input_dim, num_clusters=64, cluster_dim=128, token_dim=256)
        raise ValueError(f"Unknown aggregation: {self.aggregation_name}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        # Run frozen backbone -- no gradients needed through it
        with torch.no_grad():
            features = self.backbone(x)

        block_outputs: list[Tensor] = features["block_outputs"]
        # Slice to the blocks covered by adapters (includes the input = output
        # of the preceding block).  E.g. for large: block_outputs[8:] gives 17
        # tensors for 16 adapters.
        relevant = block_outputs[self._adapter_start:]

        if not self.hashing:
            adapted = self.side_adapter(relevant)
            return self._float_descriptor(adapted)

        if self.hashing and not self.rerank:
            adapted_h = self.side_adapter_hashing(relevant)
            z = self._hash_descriptor(adapted_h)
            return z, STEBinary.apply(z)

        # rerank: both float and hash branches
        adapted = self.side_adapter(relevant)
        adapted_h = self.side_adapter_hashing(relevant)
        x_g = self._float_descriptor(adapted)
        z = self._hash_descriptor(adapted_h)
        return z, STEBinary.apply(z), x_g

    def _tokens_to_grid(self, tokens: Tensor) -> Tensor:
        batch_size, patches, dim = tokens.shape
        width = height = int(math.sqrt(patches))
        if width * height != patches:
            raise ValueError(f"Expected a square patch grid, got {patches} tokens")
        return tokens.view(batch_size, width, height, dim).permute(0, 3, 1, 2)

    def _float_descriptor(self, adapted: Tensor) -> Tensor:
        """Produce a float descriptor from side-adapter output (CLS + patches)."""
        x_patch = adapted[:, 1:]   # drop CLS token
        x_cls = adapted[:, 0]
        x_grid = self._tokens_to_grid(x_patch)

        if self.aggregation_name == "gem":
            x_g = self.linear1(x_patch.view(x_grid.size(0), x_grid.size(2), x_grid.size(3), x_grid.size(1)))
            x_g = x_g.permute(0, 3, 1, 2)
            x_g = self.aggregation(x_g)
            x_g = self.linear2(x_g)
        elif self.aggregation_name == "boq":
            x_g = self.aggregation(x_grid)
        elif self.aggregation_name == "salad":
            x_g = self.aggregation((x_grid, x_cls))
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation_name}")

        return F.normalize(x_g, p=2, dim=-1)

    def _hash_descriptor(self, adapted: Tensor) -> Tensor:
        """Produce a hash descriptor from side-adapter-hashing output."""
        z_patch = adapted[:, 1:]
        z_grid = self._tokens_to_grid(z_patch)
        z = self.linear3(z_patch.view(z_grid.size(0), z_grid.size(2), z_grid.size(3), z_grid.size(1)))
        z = z.permute(0, 3, 1, 2)
        z = self.aggregation_hashing(z)
        z = self.linear4(z)
        return F.normalize(z, p=2, dim=-1)

    # ------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------

    def setup_for_training(self, mode: Optional[TrainMode] = None, initialize_adapters: bool = True) -> None:
        """Set ``requires_grad`` to match the SelaVPR++ training protocol."""
        if mode is None:
            if self.hashing and self.rerank:
                mode = "rerank"
            elif self.hashing:
                mode = "hashing"
            else:
                mode = "standard"

        # Freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        if mode in {"standard", "hashing"}:
            # Unfreeze side adapter(s) and all non-backbone parameters
            for name, param in self.named_parameters():
                if not name.startswith("backbone."):
                    param.requires_grad = True
            if initialize_adapters:
                self.initialize_adapters()
        elif mode == "rerank":
            # Copy float adapter weights to hashing adapter, then only train hashing branch
            if hasattr(self, "side_adapter") and hasattr(self, "side_adapter_hashing"):
                self.side_adapter_hashing.load_state_dict(self.side_adapter.state_dict())
            for name, param in self.named_parameters():
                if (
                    name.startswith("side_adapter_hashing.")
                    or name.startswith("linear3.")
                    or name.startswith("linear4.")
                    or name.startswith("aggregation_hashing.")
                ):
                    param.requires_grad = True
        else:
            raise ValueError(f"Unknown training mode: {mode}")

    def initialize_adapters(self) -> None:
        """Initialize adapter weights the same way as the SelaVPR++ training scripts."""
        targets = [self.side_adapter] if hasattr(self, "side_adapter") else []
        if hasattr(self, "side_adapter_hashing"):
            targets.append(self.side_adapter_hashing)

        for network in targets:
            for module_name, module in network.named_modules():
                if isinstance(module, MultiConvAdapter):
                    # Zero-init the up-projection so adapters start as identity
                    nn.init.constant_(module.D_fc2.weight, 0.0)
                    nn.init.constant_(module.D_fc2.bias, 0.0)
                    # Near-zero init for conv weights
                    for child_name, child in module.named_modules():
                        if isinstance(child, nn.Conv2d):
                            nn.init.constant_(child.weight, 0.00001)
                            if child.bias is not None:
                                nn.init.constant_(child.bias, 0.00001)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (param for param in self.parameters() if param.requires_grad)

    def parameter_report(self) -> dict[str, float]:
        total = sum(param.numel() for param in self.parameters()) / 1e6
        backbone = sum(param.numel() for param in self.backbone.parameters()) / 1e6
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad) / 1e6
        trainable_backbone = sum(param.numel() for param in self.backbone.parameters() if param.requires_grad) / 1e6
        return {
            "total_m": total,
            "backbone_m": backbone,
            "head_m": total - backbone,
            "trainable_m": trainable,
            "trainable_backbone_m": trainable_backbone,
        }


# ---------------------------------------------------------------------------
# Public variant classes and factory functions
# ---------------------------------------------------------------------------


class SelaVPR(SelaVPRTrainable):
    """Backward-compatible repo export for the new SelaVPR++ trainable wrapper."""

    def __init__(
        self,
        *,
        backbone: BackboneName = "dinov2-large",
        aggregation: AggregationName = "gem",
        hashing: bool = False,
        rerank: bool = False,
        foundation_model_path: Optional[str] = None,
        resume: bool = False,
        setup_training: bool = False,
        pretrained_path: Optional[str] = None,
        pretrained_url: Optional[str] = None,
        load_pretrained_backbone: bool = True,
        checkpoint_path: Optional[str] = None,
        hash_bits: Optional[int] = None,
        registers: bool = False,
        **_unused_legacy_kwargs: object,
    ) -> None:
        if registers:
            raise ValueError("SelaVPR++ does not use DINO register tokens in this wrapper")
        if hash_bits is not None and hash_bits != 512:
            raise ValueError("SelaVPR++ hashing branch is fixed at 512 dimensions")
        super().__init__(
            backbone=backbone,
            aggregation=aggregation,
            hashing=hashing or hash_bits is not None,
            rerank=rerank,
            foundation_model_path=foundation_model_path or pretrained_path,
            pretrained_url=pretrained_url,
            load_pretrained_backbone=load_pretrained_backbone,
            resume=resume,
            setup_training=setup_training,
        )
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, strict=False)

    def load_checkpoint(self, path: str, strict: bool = False) -> None:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {self._remove_prefix(k, "module."): v for k, v in state.items()}
        self.load_state_dict(state, strict=strict)

    def load_selavpr_checkpoint(self, path: str, strict: bool = False) -> None:
        self.load_checkpoint(path, strict=strict)


class _SelaVPRVariant(SelaVPR):
    backbone_name: BackboneName
    aggregation_name: AggregationName

    def __init__(self, **kwargs: object) -> None:
        backbone = kwargs.pop("backbone", self.backbone_name)
        aggregation = kwargs.pop("aggregation", self.aggregation_name)
        kwargs.setdefault("setup_training", True)
        if backbone != self.backbone_name:
            raise ValueError(f"{type(self).__name__} is fixed to backbone={self.backbone_name!r}")
        if aggregation != self.aggregation_name:
            raise ValueError(f"{type(self).__name__} is fixed to aggregation={self.aggregation_name!r}")
        super().__init__(backbone=self.backbone_name, aggregation=self.aggregation_name, **kwargs)


class SelaVPRBaseBoQ(_SelaVPRVariant):
    backbone_name = "dinov2-base"
    aggregation_name = "boq"


class SelaVPRLargeBoQ(_SelaVPRVariant):
    backbone_name = "dinov2-large"
    aggregation_name = "boq"


class SelaVPRBaseGeM(_SelaVPRVariant):
    backbone_name = "dinov2-base"
    aggregation_name = "gem"


class SelaVPRLargeGeM(_SelaVPRVariant):
    backbone_name = "dinov2-large"
    aggregation_name = "gem"


class SelaVPRBaseSALAD(_SelaVPRVariant):
    backbone_name = "dinov2-base"
    aggregation_name = "salad"


class SelaVPRLargeSALAD(_SelaVPRVariant):
    backbone_name = "dinov2-large"
    aggregation_name = "salad"


def build_selavpr_for_training(
    *,
    backbone: BackboneName = "dinov2-large",
    aggregation: AggregationName = "gem",
    foundation_model_path: Optional[str] = None,
    pretrained_url: Optional[str] = None,
    load_pretrained_backbone: bool = True,
    hashing: bool = False,
    rerank: bool = False,
    mode: Optional[TrainMode] = None,
) -> SelaVPRTrainable:
    model = SelaVPRTrainable(
        backbone=backbone,
        aggregation=aggregation,
        foundation_model_path=foundation_model_path,
        pretrained_url=pretrained_url,
        load_pretrained_backbone=load_pretrained_backbone,
        hashing=hashing,
        rerank=rerank,
    )
    model.setup_for_training(mode=mode)
    return model


def build_selavpr(
    *,
    backbone: BackboneName = "dinov2-large",
    aggregation: AggregationName = "gem",
    foundation_model_path: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    pretrained_url: Optional[str] = None,
    load_pretrained_backbone: bool = True,
    checkpoint_path: Optional[str] = None,
    hashing: bool = False,
    rerank: bool = False,
    mode: Optional[TrainMode] = None,
    setup_training: bool = False,
    hash_bits: Optional[int] = None,
    registers: bool = False,
    **legacy_kwargs: object,
) -> SelaVPR:
    model = SelaVPR(
        backbone=backbone,
        aggregation=aggregation,
        foundation_model_path=foundation_model_path or pretrained_path,
        pretrained_url=pretrained_url,
        load_pretrained_backbone=load_pretrained_backbone,
        hashing=hashing or hash_bits is not None,
        rerank=rerank,
        hash_bits=hash_bits,
        registers=registers,
        setup_training=False,
        **legacy_kwargs,
    )
    if checkpoint_path is not None:
        model.load_selavpr_checkpoint(checkpoint_path, strict=False)
    if setup_training or mode is not None:
        model.setup_for_training(mode=mode)
    return model


def build_optimizer(
    model: SelaVPRTrainable,
    *,
    optim: Literal["adam", "sgd", "adamw"] = "adam",
    lr: float = 4e-4,
    weight_decay: float = 9.5e-9,
) -> torch.optim.Optimizer:
    params = list(model.trainable_parameters())
    if optim == "adam":
        return torch.optim.Adam(params, lr=lr)
    if optim == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.001)
    if optim == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {optim}")


__all__ = [
    "DinoV2BoQ",
    "DinoV2GeM",
    "DinoV2SALAD",
    "SelaVPRBaseBoQ",
    "SelaVPRBaseGeM",
    "SelaVPRBaseSALAD",
    "SelaVPRLargeBoQ",
    "SelaVPRLargeGeM",
    "SelaVPRLargeSALAD",
]
