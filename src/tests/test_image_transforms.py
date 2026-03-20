from __future__ import annotations

import torch

from modules.transforms.image import EvalTransform, TrainTransform


def test_eval_transform_accepts_tensor_input() -> None:
    transform = EvalTransform(image_size=16)
    image = torch.randint(0, 256, (3, 20, 24), dtype=torch.uint8)

    output = transform(image)

    assert output.shape == (3, 16, 16)
    assert output.dtype == torch.float32


def test_train_transform_accepts_tensor_input() -> None:
    transform = TrainTransform(image_size=16)
    image = torch.randint(0, 256, (3, 20, 24), dtype=torch.uint8)

    output = transform(image)

    assert output.shape == (3, 16, 16)
    assert output.dtype == torch.float32
