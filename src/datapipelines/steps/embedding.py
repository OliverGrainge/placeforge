import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as T
from PIL import Image

from .base import BaseStep
from .util import EmbeddingCache


class _ImageDataset(torch.utils.data.Dataset):
    """Minimal dataset that loads and transforms images."""

    def __init__(self, paths: list[str], root: Path, transform):
        self.paths = paths
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.root / self.paths[idx]).convert("RGB")
        return self.transform(img), idx  # return idx for ordering guarantee


class ComputeImageEmbeddingStep(BaseStep):
    """
    Runs the model on all images in traindataset and writes embeddings to a
    memory-mapped array indexed directly by image_id:

        embeddings[image_id] → embedding vector

    Saved to <feature_store>/<image_embedding_name>/images.npy
    """

    TRANSFORM = T.Compose(
        [
            T.Resize((322, 322)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        image_embedding_name: str,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
        self.cache_path = (
            Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])
            / image_embedding_name
            / "images.npy"
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.cache_path.exists():
            model = self._load_model()
            self._run_model(context["traindataset"], model)
            del model
            torch.cuda.empty_cache()
        return context

    def _run_model(self, df: pd.DataFrame, model) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        image_ids = df["image_id"].tolist()
        emb_dim = self._probe_embedding_dim(model)

        mmap = np.lib.format.open_memmap(
            self.cache_path,
            mode="w+",
            dtype=np.float32,
            shape=(max(image_ids) + 1, emb_dim),
        )

        if self.pbar is not None:
            self.pbar.reset(total=len(df))

        dataset = _ImageDataset(df["image_path"].tolist(), self.raw_dir, self.TRANSFORM)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        for images, indices in loader:
            batch_ids = [image_ids[i] for i in indices.tolist()]
            with torch.no_grad():
                embs = model(images.cuda()).cpu().numpy()
            mmap[batch_ids] = embs
            if self.pbar is not None:
                self.pbar.update(len(batch_ids))

        mmap.flush()
        del mmap

    @staticmethod
    def _load_model():
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                model = torch.hub.load("serizba/salad", "dinov2_salad")
        return model.eval().cuda()

    def _probe_embedding_dim(self, model) -> int:
        dummy = torch.randn(1, 3, 322, 322, device="cuda")
        with torch.no_grad():
            return model(dummy).shape[1]


class AggregatePlaceEmbeddingStep(BaseStep):
    """
    Aggregates image embeddings into place embeddings.

    Reads image embeddings from <feature_store>/<image_embedding_name>/images.npy
    (indexed by image_id), groups by place_id from traindataset, reduces
    with `reduction`, optionally L2-normalizes, and writes results to an
    EmbeddingCache at <feature_store>/<place_embedding_name>/.
    """

    def __init__(
        self,
        image_embedding_name: str,
        place_embedding_name: str,
        reduction: Literal["mean"] = "mean",
        normalize: bool = False,
    ) -> None:
        super().__init__()
        feature_store = Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])
        self.image_cache_path = feature_store / image_embedding_name / "images.npy"
        self.place_cache = EmbeddingCache(feature_store / place_embedding_name)
        self.reduction = reduction
        self.normalize = normalize

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.place_cache.npy_path.exists():
            return context

        df = context["traindataset"]
        image_embs = np.load(self.image_cache_path, mmap_mode="r")

        grouped = list(df.groupby("place_id", sort=True))
        n_places = len(grouped)
        emb_dim = image_embs.shape[1]

        self.place_cache.cache_dir.mkdir(parents=True, exist_ok=True)
        mmap = np.lib.format.open_memmap(
            self.place_cache.npy_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_places, emb_dim),
        )

        if self.pbar is not None:
            self.pbar.reset(total=n_places)

        for place_id, (_, sub) in enumerate(grouped):
            embs = image_embs[sub["image_id"].values].astype(np.float32)

            if self.reduction == "mean":
                agg = embs.mean(axis=0)
            else:
                raise ValueError(f"Unknown reduction: {self.reduction!r}")

            if self.normalize:
                norm = np.linalg.norm(agg)
                if norm > 0:
                    agg = agg / norm

            mmap[place_id] = agg

            if self.pbar is not None:
                self.pbar.update(1)

        mmap.flush()
        del mmap

        return context
