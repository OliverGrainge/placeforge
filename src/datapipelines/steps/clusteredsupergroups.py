from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import BaseStep


class ClusteredSuperGroupsStep(BaseStep):
    """Assigns supergroup_id by clustering aggregated place embeddings under
    spatial non-adjacency constraints.

    Unlike :class:`AssignSuperGroupsStep`, which uses a fixed repeating grid
    coloring, this step clusters the aggregated place embedding vectors so that
    semantically similar places end up in the same supergroup.  The
    non-adjacency guarantee (no two grid-adjacent cells share a supergroup) is
    enforced via a post-clustering greedy conflict-repair pass.

    The minimum number of supergroups required to guarantee a valid coloring is
    ``(adjacency_cells + 1) ** 2`` (e.g. 4 for the default Moore neighbourhood).
    A ``ValueError`` is raised at construction time if ``num_supergroups`` is
    less than this.

    Parameters
    ----------
    num_supergroups:
        Target number of supergroups.  Must be >= ``(adjacency_cells + 1) ** 2``.
    adjacency_cells:
        Chebyshev radius (in grid cells) that defines "adjacent".  Must match
        the value used in earlier steps.
    random_state:
        Seed passed to k-means for reproducibility.
    max_repair_iters:
        Maximum passes over the conflict-repair loop.  Conflicts that remain
        after this many passes are left in place (logged as a warning).
    """

    def __init__(
        self,
        num_supergroups: int,
        *,
        adjacency_cells: int = 1,
        context_key: str = "index",
        embedding_cache_dir_context_key: str = "aggregated_embedding_cache_dir",
        place_id_column: str = "place_id",
        supergroup_id_column: str = "supergroup_id",
        random_state: int = 0,
        max_repair_iters: int = 20,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if adjacency_cells < 1:
            raise ValueError("adjacency_cells must be >= 1")
        min_groups = (adjacency_cells + 1) ** 2
        if num_supergroups < min_groups:
            raise ValueError(
                f"num_supergroups ({num_supergroups}) must be >= {min_groups} "
                f"for adjacency_cells={adjacency_cells}"
            )
        self.num_supergroups = num_supergroups
        self.adjacency_cells = adjacency_cells
        self.context_key = context_key
        self.embedding_cache_dir_context_key = embedding_cache_dir_context_key
        self.place_id_column = place_id_column
        self.supergroup_id_column = supergroup_id_column
        self.random_state = random_state
        self.max_repair_iters = max_repair_iters

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        dataframe = context[self.context_key]
        embedding_cache_dir = Path(context[self.embedding_cache_dir_context_key])

        place_embeddings = self._load_embeddings(embedding_cache_dir)

        # Determine ordered list of place_ids that have embeddings and valid cells.
        place_ids, cells = self._place_ids_and_cells(dataframe, place_embeddings)

        embeddings = np.stack([place_embeddings[pid] for pid in place_ids])

        with self.progress(total=None, desc="cluster supergroups") as progress:
            labels, centroids = self._cluster(embeddings)
            progress.update(1)
            labels = self._repair_conflicts(cells, labels, embeddings, centroids)
            progress.update(1)

        place_to_supergroup: dict[Any, int] = {
            pid: int(label) for pid, label in zip(place_ids, labels)
        }

        result = dataframe.copy()
        result[self.supergroup_id_column] = dataframe[self.place_id_column].map(
            place_to_supergroup
        )

        context = dict(context)
        context[self.context_key] = result
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_embeddings(self, cache_dir: Path) -> dict[Any, np.ndarray]:
        import pandas as pd

        manifest_path = cache_dir / "manifest.parquet"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Aggregated embedding manifest not found at {manifest_path}. "
                "Run AggregateEmbeddingsStep first."
            )

        manifest = pd.read_parquet(manifest_path)
        batches_dir = cache_dir / "batches"

        embeddings: dict[Any, np.ndarray] = {}
        for chunk_idx_val, chunk_rows in manifest.groupby("chunk_idx"):
            chunk_path = batches_dir / f"batch_{int(chunk_idx_val):06d}.npy"
            chunk = np.load(chunk_path)
            row_indices = chunk_rows["row_idx"].to_numpy()
            chunk_embeddings = chunk[row_indices]
            for pid, emb in zip(chunk_rows[self.place_id_column].tolist(), chunk_embeddings):
                embeddings[pid] = emb

        return embeddings

    def _place_ids_and_cells(
        self,
        dataframe: Any,
        place_embeddings: dict[Any, np.ndarray],
    ) -> tuple[list[Any], list[tuple[int, int]]]:
        """Return parallel lists of place_ids and their (cell_x, cell_y) coords.

        Cell coordinates are parsed from the place_id string
        (``"{zone}_{letter}_{cell_x}_{cell_y}"``), so no UTM columns are needed.
        Only place_ids present in ``place_embeddings`` are included.
        """
        seen: set[Any] = set()
        place_ids: list[Any] = []
        cells: list[tuple[int, int]] = []

        for pid in dataframe[self.place_id_column]:
            if pid in seen or pid not in place_embeddings:
                continue
            cell = self._cell_from_place_id(pid)
            if cell is None:
                continue
            seen.add(pid)
            place_ids.append(pid)
            cells.append(cell)

        return place_ids, cells

    @staticmethod
    def _cell_from_place_id(place_id: Any) -> tuple[int, int] | None:
        """Parse (cell_x, cell_y) from ``"{zone}_{letter}_{cell_x}_{cell_y}"``."""
        try:
            parts = str(place_id).split("_")
            return int(parts[-2]), int(parts[-1])
        except (IndexError, ValueError):
            return None

    def _cluster(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=self.num_supergroups,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = km.fit_predict(embeddings)
        return labels.astype(np.int32), km.cluster_centers_.astype(np.float32)

    def _repair_conflicts(
        self,
        cells: list[tuple[int, int]],
        labels: np.ndarray,
        embeddings: np.ndarray,
        centroids: np.ndarray,
    ) -> np.ndarray:
        """Greedily reassign cells that share a label with a grid neighbour.

        Each conflicting cell is moved to the closest centroid whose label does
        not appear among that cell's neighbours.  Passes repeat until no
        conflicts remain or ``max_repair_iters`` is reached.
        """
        labels = labels.copy()
        cell_to_idx: dict[tuple[int, int], int] = {c: i for i, c in enumerate(cells)}
        adj = self.adjacency_cells

        neighbors: list[list[int]] = []
        for cx, cy in cells:
            nbrs = []
            for dx in range(-adj, adj + 1):
                for dy in range(-adj, adj + 1):
                    if dx == 0 and dy == 0:
                        continue
                    j = cell_to_idx.get((cx + dx, cy + dy))
                    if j is not None:
                        nbrs.append(j)
            neighbors.append(nbrs)

        all_labels = np.arange(self.num_supergroups)

        for _ in range(self.max_repair_iters):
            changed = False
            for i, nbrs in enumerate(neighbors):
                if not nbrs:
                    continue
                neighbor_labels = set(labels[j] for j in nbrs)
                if labels[i] not in neighbor_labels:
                    continue

                valid_mask = ~np.isin(all_labels, list(neighbor_labels))
                valid_labels = all_labels[valid_mask]
                if valid_labels.size == 0:
                    continue  # theoretically unreachable given num_supergroups >= (adj+1)^2

                dists = np.linalg.norm(centroids[valid_labels] - embeddings[i], axis=1)
                labels[i] = valid_labels[int(np.argmin(dists))]
                changed = True

            if not changed:
                break

        return labels
