from __future__ import annotations

from typing import Any

import torch
import torchvision.transforms.v2 as transforms
from torchvision.transforms import InterpolationMode

# ImageNet mean/std, used by most pretrained vision backbones including DINOv2
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class TrainTransform:
    """Augmented transform for contrastive VPR training.

    Supports augmentation presets via the ``augmentation`` parameter:
      - ``"light"``: mild colour jitter + random crop
      - ``"heavy"``: stronger jitter, grayscale, blur, and random erasing
    """

    def __init__(
        self,
        image_size: int = 224,
        augmentation: str = "light",
    ) -> None:
        if augmentation == "light":
            aug_ops = [
                transforms.RandomResizedCrop(
                    (image_size, image_size),
                    scale=(0.7, 1.0),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1,
                ),
            ]
        elif augmentation == "heavy":
            aug_ops = [
                transforms.RandomResizedCrop(
                    (image_size, image_size),
                    scale=(0.4, 1.0),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(
                    brightness=0.7, contrast=0.7, saturation=0.7, hue=0.2,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ]
        else:
            raise ValueError(
                f"Unknown augmentation preset {augmentation!r}. "
                f"Choose 'light' or 'heavy'."
            )

        # RandomErasing operates on tensors, so it must come after ToDtype.
        # Split ops around the dtype conversion boundary.
        pre_tensor = []
        post_tensor = []
        for op in aug_ops:
            if isinstance(op, transforms.RandomErasing):
                post_tensor.append(op)
            else:
                pre_tensor.append(op)

        self._transform = transforms.Compose(
            pre_tensor
            + [
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
            + post_tensor
        )

    def __call__(self, image: Any) -> Any:
        return self._transform(image)


class EvalTransform:
    """Deterministic transform for validation and inference.

    Pipeline: Resize → ToDtype → Normalize
    """

    def __init__(self, image_size: int = 224) -> None:
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
