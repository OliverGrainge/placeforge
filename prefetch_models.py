"""Pre-download pretrained model weights into the torch cache.

Run this once on a node with internet access (e.g. login node) so that
compute nodes can load models from cache without network access.

    python prefetch_models.py
"""

import torch
from torchvision.models import resnet50, ResNet50_Weights


def main() -> None:
    print(f"TORCH_HOME = {torch.hub.get_dir()}")
    print()

    print("Downloading facebookresearch/dinov2 — dinov2_vitb14 ...")
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    print("  done.\n")

    print("Downloading torchvision ResNet-50 (ImageNet1K_V1) ...")
    resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    print("  done.\n")

    print("Downloading serizba/salad — dinov2_salad ...")
    torch.hub.load("serizba/salad", "dinov2_salad")
    print("  done.\n")

    print("All models cached.")


if __name__ == "__main__":
    main()
