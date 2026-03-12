from __future__ import annotations

from typing import Any

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from . import register_transform

# ImageNet mean/std, used by most pretrained vision backbones including DINOv2
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@register_transform("train")
class TrainTransform:
    """Standard augmented transform for training.

    Pipeline: Resize → RandomCrop → RandomHorizontalFlip → ColorJitter →
              ToTensor → Normalize
    """

    def __init__(
        self,
        image_size: int = 224,
        *,
        color_jitter_strength: float = 0.4,
        horizontal_flip_prob: float = 0.5,
    ) -> None:
        scale_size = int(image_size * 1.14)  # ~14% larger before crop
        self._transform = transforms.Compose([
            transforms.Resize(scale_size, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=horizontal_flip_prob),
            transforms.ColorJitter(
                brightness=color_jitter_strength,
                contrast=color_jitter_strength,
                saturation=color_jitter_strength,
                hue=color_jitter_strength * 0.25,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __call__(self, image: Any) -> Any:
        return self._transform(image)


@register_transform("eval")
class EvalTransform:
    """Deterministic centre-crop transform for validation and inference.

    Pipeline: Resize → CenterCrop → ToTensor → Normalize
    """

    def __init__(self, image_size: int = 224) -> None:
        scale_size = int(image_size * 1.14)
        self._transform = transforms.Compose([
            transforms.Resize(scale_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    def __call__(self, image: Any) -> Any:
        return self._transform(image)


__all__ = ["EvalTransform", "TrainTransform"]
