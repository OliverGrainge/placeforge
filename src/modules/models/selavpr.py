"""Standalone SelaVPR trainable visual place recognition model."""

from __future__ import annotations

import math
from functools import partial
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import trunc_normal_

_DINOV2_VITL14_PRETRAIN_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"


def _make_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    return (x, x)


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
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        x = self.proj(x)
        return self.proj_drop(x)


class Adapter(nn.Module):
    """SelaVPR adapter inserted into each transformer block."""

    def __init__(
        self,
        features: int,
        mlp_ratio: float = 0.75,
        act_layer: type[nn.Module] = nn.ReLU,
        skip_connect: bool = True,
    ) -> None:
        super().__init__()
        self.skip_connect = skip_connect
        hidden_features = int(features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(features, hidden_features)
        self.D_fc2 = nn.Linear(hidden_features, features)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.D_fc2(self.act(self.D_fc1(x)))
        return x + residual if self.skip_connect else residual


class Block(nn.Module):
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
        self.adapter1 = Adapter(dim, mlp_ratio=0.5)
        self.adapter2 = Adapter(dim, mlp_ratio=0.5, skip_connect=False)

    def forward(self, x: Tensor) -> Tensor:
        attn_residual = self.ls1(self.adapter1(self.attn(self.norm1(x))))
        x = x + self.drop_path1(attn_residual)
        normed = self.norm2(x)
        ffn_residual = self.ls2(self.mlp(normed) + 0.2 * self.adapter2(normed))
        return x + self.drop_path2(ffn_residual)


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
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.num_tokens = 1
        self.num_register_tokens = num_register_tokens
        self.patch_size = patch_size
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

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
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
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
        x = x + self.interpolate_pos_encoding(x, height, width)
        if self.register_tokens is not None:
            x = torch.cat((x[:, :1], self.register_tokens.expand(batch, -1, -1), x[:, 1:]), dim=1)
        return x

    def forward_features(self, x: Tensor, masks: Optional[Tensor] = None) -> Dict[str, Optional[Tensor]]:
        x = self.prepare_tokens(x, masks)
        for block in self.blocks:
            x = block(x)
        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def forward(self, x: Tensor, masks: Optional[Tensor] = None) -> Dict[str, Optional[Tensor]]:
        return self.forward_features(x, masks)


def vit_large(
    patch_size: int = 14,
    img_size: int = 518,
    num_register_tokens: int = 0,
    **kwargs,
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        img_size=img_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(x, p=2, dim=self.dim)


class LocalAdapt(nn.Module):
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 256, out_channels: int = 128) -> None:
        super().__init__()
        self.upconv1 = nn.ConvTranspose2d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.upconv1(x))
        return self.upconv2(x)


class SelaVPR(nn.Module):
    def __init__(
        self,
        registers: bool = False,
        hash_bits: Optional[int] = None,
        pretrained_path: Optional[str] = None,
        pretrained_url: str = _DINOV2_VITL14_PRETRAIN_URL,
        load_pretrained_backbone: bool = True,
        freeze_backbone_except_adapters: bool = True,
        zero_init_adapters: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = vit_large(patch_size=14, img_size=518, init_values=1.0, num_register_tokens=4 if registers else 0)
        self.aggregation = nn.Sequential(L2Norm(dim=1), GeM())
        self.local_adapt = LocalAdapt(in_channels=1024, out_channels=128)
        self.hash_head = nn.Linear(1024, hash_bits) if hash_bits is not None else None
        self.features_dim = 1024
        self.descriptor_dim = 1024

        if load_pretrained_backbone:
            self.load_pretrained_backbone(pretrained_path or pretrained_url)
        if zero_init_adapters:
            self.init_adapters_zero()
        if freeze_backbone_except_adapters:
            self.freeze_backbone_except_adapters()

    @property
    def feature_dim(self) -> int:
        return self.descriptor_dim

    @staticmethod
    def _remove_prefix(text: str, prefix: str) -> str:
        return text[len(prefix) :] if text.startswith(prefix) else text

    def freeze_backbone_except_adapters(self) -> None:
        for name, param in self.backbone.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    def init_adapters_zero(self) -> None:
        """Match SelaVPR training initialization for adapter output layers."""
        for name, module in self.named_modules():
            if "adapter" in name and name.endswith("D_fc2") and isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0.0)
                nn.init.constant_(module.bias, 0.0)

    @staticmethod
    def _load_state_dict(location: str) -> dict[str, Tensor]:
        if location.startswith(("http://", "https://")):
            state = torch.hub.load_state_dict_from_url(location, map_location="cpu", progress=True)
        else:
            state = torch.load(location, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        return state

    def load_pretrained_backbone(self, location: str, strict: bool = False) -> None:
        state = self._load_state_dict(location)
        state = {self._remove_prefix(self._remove_prefix(k, "module."), "backbone."): v for k, v in state.items()}
        current = self.backbone.state_dict()
        compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
        mismatched = [k for k, v in state.items() if k in current and current[k].shape != v.shape]
        unexpected = [k for k in state if k not in current]
        missing = [k for k in current if k not in compatible]
        current.update(compatible)
        self.backbone.load_state_dict(current, strict=strict)
        print(
            "Loaded pretrained SelaVPR backbone weights "
            f"from {location} ({len(compatible)}/{len(current)} tensors matched; "
            f"{len(missing)} missing, {len(mismatched)} shape-mismatched, {len(unexpected)} unexpected)."
        )

    def load_selavpr_checkpoint(self, path: str, strict: bool = False) -> None:
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        state = {self._remove_prefix(k, "module."): v for k, v in state.items()}
        self.load_state_dict(state, strict=strict)

    def _patch_feature(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        patch_tokens = features["x_norm_patchtokens"]
        assert isinstance(patch_tokens, Tensor)
        batch, tokens, channels = patch_tokens.shape
        grid = int(math.sqrt(tokens))
        assert grid * grid == tokens, f"Expected a square patch grid, got {tokens} tokens"
        return patch_tokens.reshape(batch, grid, grid, channels).permute(0, 3, 1, 2)

    def _global_feature(self, patch_feature: Tensor) -> Tensor:
        return F.normalize(self.aggregation(patch_feature), p=2, dim=-1)

    def _local_feature(self, patch_feature: Tensor) -> Tensor:
        local_feature = self.local_adapt(patch_feature).permute(0, 2, 3, 1)
        return F.normalize(local_feature, p=2, dim=-1)

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        patch_feature = self._patch_feature(x)
        return self._local_feature(patch_feature), self._global_feature(patch_feature)

    def forward(
        self,
        x: Tensor,
        *,
        return_local: bool = False,
        return_hash: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        patch_feature = self._patch_feature(x)
        global_feature = self._global_feature(patch_feature)

        if return_hash:
            if self.hash_head is None:
                raise RuntimeError("SelaVPR was constructed without hash_bits, so no hash head exists.")
            local_feature = self._local_feature(patch_feature)
            hash_feature = torch.tanh(self.hash_head(global_feature))
            return local_feature, global_feature, hash_feature
        if return_local:
            local_feature = self._local_feature(patch_feature)
            return local_feature, global_feature
        return global_feature


def build_selavpr(
    registers: bool = False,
    pretrained_path: Optional[str] = None,
    pretrained_url: str = _DINOV2_VITL14_PRETRAIN_URL,
    checkpoint_path: Optional[str] = None,
    hash_bits: Optional[int] = None,
    load_pretrained_backbone: bool = True,
    freeze_backbone_except_adapters: bool = True,
    zero_init_adapters: bool = True,
) -> SelaVPR:
    model = SelaVPR(
        registers=registers,
        hash_bits=hash_bits,
        pretrained_path=pretrained_path,
        pretrained_url=pretrained_url,
        load_pretrained_backbone=load_pretrained_backbone,
        freeze_backbone_except_adapters=freeze_backbone_except_adapters,
        zero_init_adapters=zero_init_adapters,
    )
    if checkpoint_path is not None:
        model.load_selavpr_checkpoint(checkpoint_path, strict=False)
    return model


__all__ = [
    "SelaVPR",
    "build_selavpr",
]
