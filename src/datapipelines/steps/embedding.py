import os
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision.transforms as T
from PIL import Image

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



class ComputeEmbeddingStep:
    """
    Computes image embeddings and aggregated place embeddings, caching both
    as memory-mapped numpy files for efficiency.

    Cache structure under <feature_store>/<name>/:
        images/embeddings.npy   — one row per image
        images/index.parquet    — image_id → row
        places/embeddings.npy   — one row per place (mean of its image embeddings)
        places/index.parquet    — place_id → row
    """

    TRANSFORM = T.Compose([
        T.Resize((322, 322)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def __init__(
        self,
        embedding_name: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pbar=None,
    ) -> None:
        self.embedding_name = embedding_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pbar = pbar
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
        self.base_dir = Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"]) / embedding_name

    @property
    def image_cache(self) -> EmbeddingCache:
        return EmbeddingCache(self.base_dir / "images")

    @property
    def place_cache(self) -> EmbeddingCache:
        return EmbeddingCache(self.base_dir / "places")

    # ---- public API ----------------------------------------------------------

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].sort_values("place_id").reset_index(drop=True)

        if not self.image_cache.exists:
            model = self._load_model()
            self._compute_image_embeddings(df, model)
            del model
            torch.cuda.empty_cache()

        if not self.place_cache.exists:
            self._compute_place_embeddings(df)

        return context

    # ---- internals -----------------------------------------------------------

    @staticmethod
    def _load_model():
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                model = torch.hub.load("serizba/salad", "dinov2_salad")
        model.eval().cuda()
        return model

    def _compute_image_embeddings(self, df: pd.DataFrame, model) -> None:
        """
        Stream image embeddings straight to a pre-allocated memory-mapped file.
        Peak RAM usage ≈ one batch of embeddings, not the full dataset.
        """
        cache = self.image_cache
        cache.cache_dir.mkdir(parents=True, exist_ok=True)

        n_images = len(df)
        image_paths = df["image_path"].tolist()
        image_ids = df["image_id"].tolist()

        if self.pbar is not None:
            self.pbar.reset(total=n_images)

        dataset = _ImageDataset(image_paths, self.raw_dir, self.TRANSFORM)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # We need the embedding dim from a single forward pass to pre-allocate.
        emb_dim = self._probe_embedding_dim(model)

        # Pre-allocate a memory-mapped file on disk — no RAM pressure.
        mmap = np.lib.format.open_memmap(
            cache.npy_path, mode="w+", dtype=np.float32, shape=(n_images, emb_dim),
        )

        for images, indices in loader:
            indices = indices.tolist()
            with torch.no_grad():
                embs = model(images.cuda()).cpu().numpy()
            mmap[indices] = embs  # write directly to disk via mmap

            if self.pbar is not None:
                self.pbar.update(len(indices))

        mmap.flush()
        del mmap

        # Write index
        pd.DataFrame({"id": image_ids, "row": range(n_images)}).to_parquet(
            cache.index_path, index=False,
        )

    def _compute_place_embeddings(self, df: pd.DataFrame) -> None:
        """
        Aggregate image embeddings per place by streaming through the
        memory-mapped image file — constant RAM regardless of dataset size.
        """
        cache = self.place_cache
        cache.cache_dir.mkdir(parents=True, exist_ok=True)

        image_embs = self.image_cache.mmap()  # memory-mapped, ~0 RAM

        # Group row indices by place_id (the df is already sorted by place_id)
        grouped = df.groupby("place_id", sort=True).indices  # {place_id: array of row indices}
        place_ids = sorted(grouped.keys())
        n_places = len(place_ids)
        emb_dim = image_embs.shape[1]

        mmap = np.lib.format.open_memmap(
            cache.npy_path, mode="w+", dtype=np.float32, shape=(n_places, emb_dim),
        )

        if self.pbar is not None:
            self.pbar.reset(total=n_places)

        for place_row, place_id in enumerate(place_ids):
            rows = grouped[place_id]
            # For places with many images, compute the mean in chunks to avoid
            # pulling a huge slice into RAM at once.
            place_sum = np.zeros(emb_dim, dtype=np.float64)
            chunk_size = 512
            for start in range(0, len(rows), chunk_size):
                chunk_rows = rows[start : start + chunk_size]
                place_sum += image_embs[chunk_rows].astype(np.float64).sum(axis=0)
            mmap[place_row] = (place_sum / len(rows)).astype(np.float32)

            if self.pbar is not None:
                self.pbar.update(1)

        mmap.flush()
        del mmap

        pd.DataFrame({"id": place_ids, "row": range(n_places)}).to_parquet(
            cache.index_path, index=False,
        )

    def _probe_embedding_dim(self, model) -> int:
        """Run a single dummy image through the model to discover embedding dimensionality."""
        dummy = torch.randn(1, 3, 322, 322, device="cuda")
        with torch.no_grad():
            return model(dummy).shape[1]