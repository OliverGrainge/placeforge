from __future__ import annotations

from math import floor
from typing import Any
import os

from pathlib import Path
import numpy as np
import pandas as pd
import torch 

from .base import BaseStep
from .util import EmbeddingCache


class AssignPlaceIdStep(BaseStep):
    def __init__(self, cell_size_meters: float) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()
        cell_size = self.cell_size_meters

        df["cell_x"] = (df["utm_east"] / cell_size).apply(floor)
        df["cell_y"] = (df["utm_north"] / cell_size).apply(floor)
        df["place_id"] = df["cell_x"] * (df["cell_y"].max() + 1) + df["cell_y"]
        df = df.sort_values("place_id").reset_index(drop=True)

        return {**context, "traindataset": df}



class AssignPlaceIdWithEmbedStep(BaseStep):
    """
    Iteratively remove incoherent images from each place.
 
    For each place:
      1. L2-normalise the image embeddings.
      2. Compute each image's mean cosine similarity to the other images.
      3. If the lowest mean similarity is below `cos_sim_threshold`, drop that
         image and repeat from step 2.
      4. Stop when every remaining image is coherent.
      5. If the number of surviving images is below `min_images`, remove the
         entire place from the dataset.
 
    When a CUDA GPU is available, places are processed in padded batches
    entirely on-device, avoiding per-place kernel launch overhead.
    """
 
    def __init__(
        self,
        embedding_name: str,
        cos_sim_threshold: float = 0.3,
        min_images: int = 2,
        gpu_batch_places: int = 512,
    ) -> None:
        super().__init__()
        self.cos_sim_threshold = cos_sim_threshold
        self.min_images = min_images
        self.gpu_batch_places = gpu_batch_places
        feature_dir = Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"]) / embedding_name
        self.image_cache = EmbeddingCache(feature_dir / "images")
 
    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
 
    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["traindataset"].copy()
 
        cache_index = self.image_cache.load_index().set_index("id")
        image_embs = self.image_cache.mmap()
 
        # Build per-place groups: list of (place_id, df_indices, cache_rows)
        groups: list[tuple[int, list[int], np.ndarray]] = []
        for place_id, sub in df.groupby("place_id"):
            df_indices = sub.index.tolist()
            image_ids = sub["image_id"].tolist()
            rows = cache_index.loc[image_ids, "row"].values
            groups.append((place_id, df_indices, rows))
 
        if self.pbar is not None:
            self.pbar.reset(total=len(groups))
 
        use_gpu = torch.cuda.is_available()
        keep_mask = np.ones(len(df), dtype=bool)
 
        if use_gpu:
            self._run_gpu(groups, image_embs, keep_mask)
        else:
            self._run_cpu(groups, image_embs, keep_mask)
 
        df = df.loc[keep_mask].reset_index(drop=True)
        return {**context, "traindataset": df}
 
    # ------------------------------------------------------------------
    # GPU path — batched across places
    # ------------------------------------------------------------------
 
    def _run_gpu(
        self,
        groups: list[tuple[int, list[int], np.ndarray]],
        image_embs: np.ndarray,
        keep_mask: np.ndarray,
    ) -> None:
        for start in range(0, len(groups), self.gpu_batch_places):
            batch = groups[start : start + self.gpu_batch_places]
            self._process_gpu_batch(batch, image_embs, keep_mask)
 
    @torch.no_grad()
    def _process_gpu_batch(
        self,
        batch: list[tuple[int, list[int], np.ndarray]],
        image_embs: np.ndarray,
        keep_mask: np.ndarray,
    ) -> None:
        B = len(batch)
        max_k = max(len(rows) for _, _, rows in batch)
        emb_dim = image_embs.shape[1]
 
        # Build padded tensor (B, max_k, D) and alive mask (B, max_k)
        padded = torch.zeros(B, max_k, emb_dim, device="cuda")
        alive = torch.zeros(B, max_k, dtype=torch.bool, device="cuda")
 
        df_indices_map: list[list[int]] = []
 
        for i, (place_id, df_indices, rows) in enumerate(batch):
            k = len(rows)
            embs = torch.from_numpy(image_embs[rows].astype(np.float32)).cuda()
            padded[i, :k] = F.normalize(embs, dim=-1)
            alive[i, :k] = True
            df_indices_map.append(df_indices)
 
        threshold = self.cos_sim_threshold
        min_images = self.min_images
 
        # Pre-compute diagonal mask once
        diag_mask = torch.eye(max_k, dtype=torch.bool, device="cuda").unsqueeze(0)
 
        # Iterative eviction — fully on GPU, vectorised across all places
        # No floor: keep evicting until every surviving image is coherent
        while True:
            alive_counts = alive.sum(dim=1)  # (B,)
 
            # Pairwise cosine similarity: (B, max_k, max_k)
            sim = torch.bmm(padded, padded.transpose(1, 2))
 
            # Mask: only pairs where both images are alive, exclude diagonal
            pair_alive = alive.unsqueeze(2) & alive.unsqueeze(1)
            valid = pair_alive & ~diag_mask
 
            sim = sim * valid.float()
 
            # Mean sim per image (over valid pairs only)
            valid_count = valid.sum(dim=2).clamp(min=1)
            mean_sim = sim.sum(dim=2) / valid_count.float()
 
            # Dead slots → inf so they're never picked as worst
            mean_sim.masked_fill_(~alive, float("inf"))
 
            worst_sim, worst_idx = mean_sim.min(dim=1)  # (B,), (B,)
 
            # Evict if below threshold AND place still has >1 image
            # (can't compute similarity with a single image)
            needs_eviction = (worst_sim < threshold) & (alive_counts > 1)
 
            if not needs_eviction.any():
                break
 
            evict_b = needs_eviction.nonzero(as_tuple=True)[0]
            evict_k = worst_idx[evict_b]
            alive[evict_b, evict_k] = False
 
        # Write results back to keep_mask
        # If a place has fewer than min_images survivors, drop the entire place
        for i, (place_id, df_indices, rows) in enumerate(batch):
            k = len(rows)
            alive_cpu = alive[i, :k].cpu().numpy()
            n_surviving = int(alive_cpu.sum())
 
            if n_surviving < min_images:
                # Remove entire place
                for local_idx in range(k):
                    keep_mask[df_indices[local_idx]] = False
            else:
                # Remove only the pruned images
                for local_idx in range(k):
                    if not alive_cpu[local_idx]:
                        keep_mask[df_indices[local_idx]] = False
 
            if self.pbar is not None:
                self.pbar.update(1)
 
    # ------------------------------------------------------------------
    # CPU fallback — per-place iteration
    # ------------------------------------------------------------------
 
    def _run_cpu(
        self,
        groups: list[tuple[int, list[int], np.ndarray]],
        image_embs: np.ndarray,
        keep_mask: np.ndarray,
    ) -> None:
        for place_id, df_indices, rows in groups:
            embeds = image_embs[rows].astype(np.float32)
            dropped = self._filter_place_cpu(embeds)
            n_surviving = len(rows) - len(dropped)
 
            if n_surviving < self.min_images:
                # Remove entire place
                for local_idx in range(len(rows)):
                    keep_mask[df_indices[local_idx]] = False
            else:
                # Remove only the pruned images
                for local_idx in dropped:
                    keep_mask[df_indices[local_idx]] = False
 
            if self.pbar is not None:
                self.pbar.update(1)
 
    def _filter_place_cpu(self, embeds: np.ndarray) -> set[int]:
        n = len(embeds)
        if n <= 1:
            return set()
 
        alive = list(range(n))
        dropped: set[int] = set()
 
        norms = np.linalg.norm(embeds, axis=-1, keepdims=True)
        normed = embeds / np.maximum(norms, 1e-8)
 
        while len(alive) > 1:
            subset = normed[alive]
            sim = subset @ subset.T
            k = len(alive)
            np.fill_diagonal(sim, 0.0)
            mean_sim = sim.sum(axis=1) / (k - 1)
 
            worst = int(np.argmin(mean_sim))
            if mean_sim[worst] >= self.cos_sim_threshold:
                break
 
            dropped.add(alive[worst])
            alive.pop(worst)
 
        return dropped

    







    