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

    Saved to <feature_store>/embedding/image/<source>/
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
        source: str,
        batch_size: int = 32,
        num_workers: int = 0,
        compile: bool = True,
        half_precision: bool = True,
    ) -> None:
        super().__init__()
        self.source = source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.compile = compile
        self.half_precision = half_precision
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
        self.image_cache = EmbeddingCache(
            Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])
            / "embedding"
            / "image"
            / source
        )

    def cache_params(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "cache_dir": str(self.image_cache.cache_dir),
            "half_precision": self.half_precision,
        }

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.image_cache.exists:
            df = context["traindataset"]
            df = df[df["source"] == self.source].reset_index(drop=True)
            model = self._load_model(self.compile)
            self._run_model(df, model)
            del model
            torch.cuda.empty_cache()
        return context

    def _run_model(self, df: pd.DataFrame, model) -> None:
        self.image_cache.cache_dir.mkdir(parents=True, exist_ok=True)

        image_paths = df["image_path"].tolist()
        sorted_paths = sorted(set(image_paths))
        path_to_row = {p: row for row, p in enumerate(sorted_paths)}
        n_images = len(sorted_paths)
        emb_dim = self._probe_embedding_dim(model)

        mmap = np.lib.format.open_memmap(
            self.image_cache.npy_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_images, emb_dim),
        )

        if self.pbar is not None:
            self.pbar.reset(total=n_images)

        dataset = _ImageDataset(image_paths, self.raw_dir, self.TRANSFORM)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

        autocast_dtype = torch.float16 if self.half_precision else torch.float32
        for images, indices in loader:
            batch_paths = [image_paths[i] for i in indices.tolist()]
            batch_rows = [path_to_row[p] for p in batch_paths]
            images = images.cuda(non_blocking=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
                embs = model(images).float().cpu().numpy()
            mmap[batch_rows] = embs
            if self.pbar is not None:
                self.pbar.update(len(batch_paths))

        mmap.flush()
        del mmap

        pd.DataFrame({"id": sorted_paths, "row": np.arange(n_images)}).to_parquet(
            self.image_cache.index_path, index=False
        )

    @staticmethod
    def _load_model(compile: bool = True):
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                model = torch.hub.load("serizba/salad", "dinov2_salad")
        model = model.eval().cuda()
        if compile:
            model = torch.compile(model)
        return model

    def _probe_embedding_dim(self, model) -> int:
        dummy = torch.randn(1, 3, 322, 322, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16 if self.half_precision else torch.float32):
            return model(dummy).float().shape[1]


class AggregatePlaceEmbeddingStep(BaseStep):
    """
    Aggregates image embeddings into place embeddings.

    Reads per-source image embeddings from <feature_store>/embedding/image/<source>/,
    groups by place_id from traindataset, reduces with `reduction`, optionally
    L2-normalizes, and writes results to an EmbeddingCache at
    <feature_store>/embedding/place/<place_embedding_name>/.
    """

    def __init__(
        self,
        place_embedding_name: str,
        reduction: Literal["mean"] = "mean",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.feature_store = Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"])
        self.place_cache = EmbeddingCache(
            self.feature_store / "embedding" / "place" / place_embedding_name
        )
        self.reduction = reduction
        self.normalize = normalize

    def _image_cache(self, source: str) -> EmbeddingCache:
        return EmbeddingCache(
            self.feature_store / "embedding" / "image" / source
        )

    def cache_params(self) -> dict[str, Any]:
        return {
            "place_cache_dir": str(self.place_cache.cache_dir),
            "reduction": self.reduction,
            "normalize": self.normalize,
        }

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"]

        # Load per-source embedding caches (memory-mapped)
        sources = sorted(df["source"].unique())
        source_to_idx = {s: i for i, s in enumerate(sources)}
        emb_arrays = []
        for source in sources:
            cache = self._image_cache(source)
            emb_arrays.append(cache.mmap())

        emb_dim = emb_arrays[0].shape[1]

        # Pre-resolve all image paths → (source_index, embedding_row)
        src_idx = np.array(
            [source_to_idx[s] for s in df["source"].values], dtype=np.int32,
        )
        emb_row = np.empty(len(df), dtype=np.int64)
        for source in sources:
            cache = self._image_cache(source)
            idx = cache.load_index().set_index("id")["row"]
            mask = df["source"].values == source
            emb_row[mask] = idx.loc[df.loc[mask, "image_path"].values].values

        # Build groups with numpy (no pandas in hot loop)
        place_ids = df["place_id"].values
        order = np.argsort(place_ids, kind="mergesort")
        sorted_pids = place_ids[order]
        split_points = np.flatnonzero(np.diff(sorted_pids)) + 1
        groups = np.split(order, split_points)
        unique_place_ids = sorted_pids[np.r_[0, split_points]]

        n_places = len(groups)
        self.place_cache.cache_dir.mkdir(parents=True, exist_ok=True)
        mmap = np.lib.format.open_memmap(
            self.place_cache.npy_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_places, emb_dim),
        )

        if self.pbar is not None:
            self.pbar.reset(total=n_places)

        for row, group_indices in enumerate(groups):
            g_src = src_idx[group_indices]
            g_rows = emb_row[group_indices]

            unique_sources = np.unique(g_src)
            if len(unique_sources) == 1:
                embs = emb_arrays[unique_sources[0]][g_rows].astype(np.float32)
            else:
                parts = []
                for si in unique_sources:
                    mask = g_src == si
                    parts.append(emb_arrays[si][g_rows[mask]].astype(np.float32))
                embs = np.concatenate(parts, axis=0)

            if self.reduction == "mean":
                agg = embs.mean(axis=0)
            else:
                raise ValueError(f"Unknown reduction: {self.reduction!r}")

            if self.normalize:
                norm = np.linalg.norm(agg)
                if norm > 0:
                    agg = agg / norm

            mmap[row] = agg

            if self.pbar is not None:
                self.pbar.update(1)

        mmap.flush()
        del mmap

        pd.DataFrame({
            "id": unique_place_ids,
            "row": np.arange(n_places),
        }).to_parquet(self.place_cache.index_path, index=False)

        return context
