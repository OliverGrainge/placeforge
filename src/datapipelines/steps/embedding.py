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
    Runs the model on all images in traindataset and writes embeddings to an
    EmbeddingCache (compact mmap + index.parquet mapping image_id → row).

    Saved to <feature_store>/<image_embedding_name>/
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
        self.image_cache = EmbeddingCache(
            Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])
            / "embedding"
            / "image"
            / image_embedding_name
        )

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.image_cache.exists:
            model = self._load_model()
            self._run_model(context["traindataset"], model)
            del model
            torch.cuda.empty_cache()
        return context

    def _run_model(self, df: pd.DataFrame, model) -> None:
        self.image_cache.cache_dir.mkdir(parents=True, exist_ok=True)

        sorted_image_ids = sorted(df["image_id"].unique().tolist())
        id_to_row = {id_: row for row, id_ in enumerate(sorted_image_ids)}
        n_images = len(sorted_image_ids)
        emb_dim = self._probe_embedding_dim(model)

        mmap = np.lib.format.open_memmap(
            self.image_cache.npy_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_images, emb_dim),
        )

        if self.pbar is not None:
            self.pbar.reset(total=n_images)

        image_ids = df["image_id"].tolist()
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
            batch_rows = [id_to_row[id_] for id_ in batch_ids]
            with torch.no_grad():
                embs = model(images.cuda()).cpu().numpy()
            mmap[batch_rows] = embs
            if self.pbar is not None:
                self.pbar.update(len(batch_ids))

        mmap.flush()
        del mmap

        pd.DataFrame({"id": sorted_image_ids, "row": np.arange(n_images)}).to_parquet(
            self.image_cache.index_path, index=False
        )

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
        self.image_cache = EmbeddingCache(
            feature_store / "embedding" / "image" / image_embedding_name
        )
        self.place_cache = EmbeddingCache(
            feature_store / "embedding" / "place" / place_embedding_name
        )
        self.reduction = reduction
        self.normalize = normalize

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.place_cache.exists:
            return context

        df = context["traindataset"]
        image_embs = self.image_cache.mmap()
        image_index = self.image_cache.load_index().set_index("id")["row"]

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

        index_ids = []
        for row, (actual_place_id, sub) in enumerate(grouped):
            rows = image_index.loc[sub["image_id"].values].values
            embs = image_embs[rows].astype(np.float32)

            if self.reduction == "mean":
                agg = embs.mean(axis=0)
            else:
                raise ValueError(f"Unknown reduction: {self.reduction!r}")

            if self.normalize:
                norm = np.linalg.norm(agg)
                if norm > 0:
                    agg = agg / norm

            mmap[row] = agg
            index_ids.append(actual_place_id)

            if self.pbar is not None:
                self.pbar.update(1)

        mmap.flush()
        del mmap

        pd.DataFrame({"id": index_ids, "row": np.arange(n_places)}).to_parquet(
            self.place_cache.index_path, index=False
        )

        return context
