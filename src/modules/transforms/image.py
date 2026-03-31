from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode

from . import register_transform

# ImageNet mean/std, used by most pretrained vision backbones including DINOv2
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@register_transform("train")
class TrainTransform:
    """Augmented transform for contrastive VPR training.

    Pipeline: RandomResizedCrop → ColorJitter → ToDtype → Normalize
    """

    def __init__(
        self,
        image_size: int = 224,
    ) -> None:
        self._transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (image_size, image_size),
                    scale=(0.5, 1.0),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(
                    brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2,
                ),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def __call__(self, image: Any) -> Any:
        return self._transform(image)


@register_transform("eval")
class EvalTransform:
    """Deterministic centre-crop transform for validation and inference.

    Pipeline: Resize → CenterCrop → ToDtype → Normalize
    """

    def __init__(self, image_size: int = 224) -> None:
        scale_size = int(image_size * 1.14)
        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )

    def __call__(self, image: Any) -> Any:
        return self._transform(image)


__all__ = ["EvalTransform", "TrainTransform"]
