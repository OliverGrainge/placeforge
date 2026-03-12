from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseStep

# DINOv2 model output dimensions (CLS token)
_DINOV2_EMBED_DIMS: dict[str, int] = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
    "dinov2_vits14_reg": 384,
    "dinov2_vitb14_reg": 768,
    "dinov2_vitl14_reg": 1024,
    "dinov2_vitg14_reg": 1536,
}

_CHUNK_SIZE = 1024  # images per batch file


class ExtractEmbeddingsStep(BaseStep):
    """Extract DINOv2 embeddings for all images and cache them on disk.

    Embeddings are stored as chunked ``.npy`` files under ``{cache_dir}/batches/``
    with a ``manifest.parquet`` index mapping each image_id to its chunk and
    row.  Already-cached images are skipped so the step is fully resumable
    after interruption.

    The step adds ``embedding_cache_dir`` and ``embedding_dim`` to the context
    for downstream steps.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        model_name: str = "dinov2_vitb14_reg",
        batch_size: int = 64,
        image_size: int = 224,
        device: str | None = None,
        context_key: str = "index",
        image_path_column: str = "image_path",
        image_id_column: str = "image_id",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if model_name not in _DINOV2_EMBED_DIMS:
            raise ValueError(
                f"Unknown model {model_name!r}. "
                f"Supported: {sorted(_DINOV2_EMBED_DIMS)}"
            )
        self.cache_dir = Path(cache_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.device = device
        self.context_key = context_key
        self.image_path_column = image_path_column
        self.image_id_column = image_id_column

    @property
    def embed_dim(self) -> int:
        return _DINOV2_EMBED_DIMS[self.model_name]

    @property
    def _batches_dir(self) -> Path:
        return self.cache_dir / "batches"

    @property
    def _manifest_path(self) -> Path:
        return self.cache_dir / "manifest.parquet"

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        import pandas as pd

        dataframe = context[self.context_key]
        image_ids: list[str] = dataframe[self.image_id_column].tolist()
        image_paths: list[str] = dataframe[self.image_path_column].tolist()

        cached_ids = self._load_cached_ids()
        uncached = [
            (iid, ipath)
            for iid, ipath in zip(image_ids, image_paths)
            if iid not in cached_ids
        ]

        if uncached:
            self._extract_and_cache(uncached)

        context = dict(context)
        context["embedding_cache_dir"] = self.cache_dir
        context["embedding_dim"] = self.embed_dim
        return context

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cached_ids(self) -> set[str]:
        if not self._manifest_path.exists():
            return set()
        import pandas as pd

        manifest = pd.read_parquet(self._manifest_path)
        return set(manifest["image_id"].tolist())

    def _next_chunk_idx(self) -> int:
        existing = sorted(self._batches_dir.glob("batch_*.npy"))
        return len(existing)

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_and_cache(self, uncached: list[tuple[str, str]]) -> None:
        import pandas as pd

        self._batches_dir.mkdir(parents=True, exist_ok=True)

        device = self._resolve_device()
        model = self._load_model(device)
        transform = self._build_transform()

        # Accumulate new manifest rows; append to existing manifest after each chunk
        chunk_idx = self._next_chunk_idx()
        pending_manifest: list[dict[str, Any]] = []

        with self.progress(total=len(uncached), desc="extract embeddings") as progress:
            for batch_start in range(0, len(uncached), self.batch_size):
                batch = uncached[batch_start : batch_start + self.batch_size]
                batch_ids = [item[0] for item in batch]
                batch_paths = [item[1] for item in batch]

                embeddings = self._embed_batch(batch_paths, model, transform, device)
                # Each GPU batch gets its own .npy file; chunk_idx always advances.
                self._save_chunk(chunk_idx, embeddings, batch_ids, pending_manifest)
                chunk_idx += 1

                # Flush manifest to disk periodically so interrupted runs can resume.
                is_last = batch_start + self.batch_size >= len(uncached)
                if len(pending_manifest) >= _CHUNK_SIZE or is_last:
                    self._flush_manifest(pending_manifest)
                    pending_manifest = []

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

        # DINOv2 returns CLS token as a plain tensor when called directly
        return out.cpu().numpy().astype(np.float32)

    def _save_chunk(
        self,
        chunk_idx: int,
        embeddings: Any,
        image_ids: list[str],
        pending_manifest: list[dict[str, Any]],
    ) -> None:
        chunk_path = self._batches_dir / f"batch_{chunk_idx:06d}.npy"
        np.save(chunk_path, embeddings)
        for row_idx, image_id in enumerate(image_ids):
            pending_manifest.append(
                {"image_id": image_id, "chunk_idx": chunk_idx, "row_idx": row_idx}
            )

    def _flush_manifest(self, new_rows: list[dict[str, Any]]) -> None:
        import pandas as pd

        new_df = pd.DataFrame(new_rows)
        if self._manifest_path.exists():
            existing = pd.read_parquet(self._manifest_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_parquet(self._manifest_path, index=False)

    # ------------------------------------------------------------------
    # Model / transform
    # ------------------------------------------------------------------

    def _load_model(self, device: Any) -> Any:
        import torch

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
