from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseStep

# Supported embedding models and their output dimensions.
_EMBED_DIMS: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
    "dinov2_salad": 8448,
}

class ExtractEmbeddingsStep(BaseStep):
    """Extract image embeddings for all images and cache them on disk.

    Embeddings are stored as chunked ``.npy`` files under ``{cache_dir}/batches/``
    with a ``manifest.parquet`` index mapping each image_id to its chunk and
    row.  Already-cached images are skipped so the step is fully resumable
    after interruption.

    The step adds ``embedding_cache_dir`` and ``embedding_dim`` to the context
    for downstream steps, along with ``embedding_index_path`` pointing at the
    manifest/index file.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        model_name: str = "dinov2_salad",
        batch_size: int = 64,
        image_size: int = 322,
        device: str | None = None,
        context_key: str = "index",
        image_path_column: str = "image_path",
        image_id_column: str = "image_id",
        cache_dir_context_key: str | None = None,
        output_cache_dir_context_key: str = "embedding_cache_dir",
        output_index_path_context_key: str = "embedding_index_path",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if model_name not in _EMBED_DIMS:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Supported: {sorted(_EMBED_DIMS)}"
            )
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.context_key = context_key
        self.image_path_column = image_path_column
        self.image_id_column = image_id_column
        self.cache_dir_context_key = cache_dir_context_key
        self.output_cache_dir_context_key = output_cache_dir_context_key
        self.output_index_path_context_key = output_index_path_context_key

    @property
    def embed_dim(self) -> int:
        return _EMBED_DIMS[self.model_name]

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        cache_dir = self._resolve_cache_dir(context)
        dataframe = context[self.context_key]
        image_ids: list[str] = dataframe[self.image_id_column].tolist()
        image_paths: list[str] = dataframe[self.image_path_column].tolist()

        cached_ids = self._load_cached_ids(cache_dir)
        completed_images = sum(1 for image_id in image_ids if image_id in cached_ids)
        uncached = [
            (iid, ipath)
            for iid, ipath in zip(image_ids, image_paths)
            if iid not in cached_ids
        ]

        if uncached:
            self._extract_and_cache(
                uncached,
                cache_dir,
                total_images=len(image_ids),
                completed_images=completed_images,
            )

        context = dict(context)
        context[self.output_cache_dir_context_key] = cache_dir
        context[self.output_index_path_context_key] = self._manifest_path(cache_dir)
        context["embedding_dim"] = self.embed_dim
        return context

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _resolve_cache_dir(self, context: dict[str, Any]) -> Path:
        if self.cache_dir_context_key and self.cache_dir_context_key in context:
            return Path(context[self.cache_dir_context_key])
        return self.cache_dir

    def _batches_dir(self, cache_dir: Path) -> Path:
        return cache_dir / "batches"

    def _manifest_path(self, cache_dir: Path) -> Path:
        return cache_dir / "manifest.parquet"

    def _load_cached_ids(self, cache_dir: Path) -> set[str]:
        manifest = self._load_manifest(cache_dir)
        if manifest is None:
            return set()
        return set(manifest["image_id"].tolist())

    def _load_manifest(self, cache_dir: Path) -> Any:
        import pandas as pd

        manifest_path = self._manifest_path(cache_dir)
        if not manifest_path.exists():
            return None

        manifest = pd.read_parquet(manifest_path)
        if manifest.empty:
            return manifest

        batch_dir = self._batches_dir(cache_dir)
        valid_chunks = {
            int(path.stem.split("_")[1])
            for path in batch_dir.glob("batch_*.npy")
        }
        repaired = manifest[manifest["chunk_idx"].isin(valid_chunks)].reset_index(drop=True)
        if len(repaired) != len(manifest):
            self._write_manifest(cache_dir, repaired)
        return repaired

    def _next_chunk_idx(self, cache_dir: Path) -> int:
        max_idx = -1
        for path in self._batches_dir(cache_dir).glob("batch_*.npy"):
            try:
                chunk_idx = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            max_idx = max(max_idx, chunk_idx)
        return max_idx + 1

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_and_cache(
        self,
        uncached: list[tuple[str, str]],
        cache_dir: Path,
        *,
        total_images: int,
        completed_images: int,
    ) -> None:
        self._batches_dir(cache_dir).mkdir(parents=True, exist_ok=True)

        # Suppress DINOv2/SALAD xFormers warnings when the optimized kernels are absent.
        warnings.filterwarnings("ignore", message=".*xFormers is not available.*")

        device = self._resolve_device()
        model = self._load_model(device)
        transform = self._build_transform()

        chunk_idx = self._next_chunk_idx(cache_dir)

        with self.progress(
            total=total_images,
            initial=completed_images,
            desc="extract embeddings",
        ) as progress:
            for batch_start in range(0, len(uncached), self.batch_size):
                batch = uncached[batch_start : batch_start + self.batch_size]
                batch_ids = [item[0] for item in batch]
                batch_paths = [item[1] for item in batch]

                embeddings = self._embed_batch(batch_paths, model, transform, device)
                manifest_rows = self._save_chunk(cache_dir, chunk_idx, embeddings, batch_ids)
                self._flush_manifest(cache_dir, manifest_rows)
                chunk_idx += 1

                progress.update(len(batch))

    def _embed_batch(
        self,
        image_paths: list[str],
        model: Any,
        transform: Any,
        device: Any,
    ) -> Any:
        import torch
        from PIL import Image

        tensors = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensors.append(transform(img))
            except Exception:
                tensors.append(torch.zeros(3, self.image_size, self.image_size))

        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            out = model(batch_tensor)

        if isinstance(out, tuple):
            out = out[0]

        return out.cpu().numpy().astype(np.float32)

    def _save_chunk(
        self,
        cache_dir: Path,
        chunk_idx: int,
        embeddings: Any,
        image_ids: list[str],
    ) -> list[dict[str, Any]]:
        chunk_path = self._batches_dir(cache_dir) / f"batch_{chunk_idx:06d}.npy"
        tmp_path = chunk_path.with_suffix(".tmp.npy")
        np.save(tmp_path, embeddings)
        tmp_path.replace(chunk_path)
        return [
            {"image_id": image_id, "chunk_idx": chunk_idx, "row_idx": row_idx}
            for row_idx, image_id in enumerate(image_ids)
        ]

    def _flush_manifest(self, cache_dir: Path, new_rows: list[dict[str, Any]]) -> None:
        import pandas as pd

        new_df = pd.DataFrame(new_rows)
        manifest_path = self._manifest_path(cache_dir)
        if manifest_path.exists():
            existing = pd.read_parquet(manifest_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        self._write_manifest(cache_dir, new_df)

    def _write_manifest(self, cache_dir: Path, dataframe: Any) -> None:
        manifest_path = self._manifest_path(cache_dir)
        tmp_path = manifest_path.with_suffix(".tmp.parquet")
        dataframe.to_parquet(tmp_path, index=False)
        tmp_path.replace(manifest_path)

    # ------------------------------------------------------------------
    # Model / transform
    # ------------------------------------------------------------------

    def _load_model(self, device: Any) -> Any:
        import torch

        if self.model_name == "dinov2_salad":
            model = torch.hub.load(
                "serizba/salad",
                "dinov2_salad",
            )
        else:
            model = torch.hub.load(
                "facebookresearch/dinov2",
                self.model_name,
                verbose=False,
            )
        model.eval()
        model.to(device)
        return model

    def _build_transform(self) -> Any:
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _resolve_device(self) -> Any:
        import torch

        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
