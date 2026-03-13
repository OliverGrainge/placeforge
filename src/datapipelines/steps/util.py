from pathlib import Path

import numpy as np
import pandas as pd



class EmbeddingCache:
    """
    Memory-mapped embedding store backed by a single .npy file + a parquet index.

    Layout on disk:
        <cache_dir>/
            embeddings.npy   — shape (N, D), float32, memory-mapped
            index.parquet     — columns: [id, row]  (id → row in the .npy)
    """

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.npy_path = cache_dir / "embeddings.npy"
        self.index_path = cache_dir / "index.parquet"

    @property
    def exists(self) -> bool:
        return self.npy_path.exists() and self.index_path.exists()

    def load_index(self) -> pd.DataFrame:
        return pd.read_parquet(self.index_path)

    def mmap(self, mode: str = "r") -> np.ndarray:
        """Return a memory-mapped view of the embeddings (no RAM cost)."""
        return np.load(self.npy_path, mmap_mode=mode)

    def lookup(self, ids: list[int]) -> np.ndarray:
        """Fetch a small number of embeddings by id into RAM."""
        index = self.load_index().set_index("id")
        rows = index.loc[ids, "row"].values
        embs = self.mmap()
        return embs[rows]