from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class VPRValidationDataset:
    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        image_root: str | Path | None = None,
        transform: Any = None,
        load_images: bool = True,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.image_root = Path(image_root) if image_root is not None else None
        self.transform = transform
        self.load_images = load_images

        self.queries_path = self.dataset_dir / "queries.parquet"
        self.database_path = self.dataset_dir / "database.parquet"
        self.matches_path = self.dataset_dir / "matches.parquet"
        self.metadata_path = self.dataset_dir / "metadata.json"

        self._validate_layout()

        queries = self._read_split(self.queries_path, id_column="qid")
        database = self._read_split(self.database_path, id_column="dbid")
        matches = self._read_matches()

        self.metadata = self._read_metadata()
        self.database_records = self._build_database_records(database)
        self.query_id_to_positive_dbids = self._build_positive_index(matches)
        self.query_records = self._build_query_records(queries, database_offset=len(self.database_records))
        self.records = [*self.database_records, *self.query_records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        if self.load_images:
            image = self._load_image(record["image_path"])
            if self.transform is not None:
                image = self.transform(image)
            record["image"] = image
        return record

    @property
    def num_queries(self) -> int:
        return len(self.query_records)

    @property
    def num_database(self) -> int:
        return len(self.database_records)

    def ground_truth(self) -> list[tuple[int, list[int]]]:
        """Return query-to-database relevance as dataset-local indices.

        Output format:
            [(query_dataset_index, [database_dataset_index, ...]), ...]
        """
        return [
            (int(record["dataset_index"]), list(record["positive_dbids"]))
            for record in self.query_records
        ]

    def _validate_layout(self) -> None:
        required_paths = (
            self.queries_path,
            self.database_path,
            self.matches_path,
            self.metadata_path,
        )
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Validation dataset is missing required files: " + ", ".join(missing_paths)
            )

    def _read_split(self, split_path: Path, *, id_column: str) -> "pd.DataFrame":
        pd = self._import_pandas()
        dataframe = pd.read_parquet(split_path)
        required_columns = (id_column, "path", "lat", "lon")
        missing_columns = [column for column in required_columns if column not in dataframe.columns]
        if missing_columns:
            raise KeyError(f"{split_path} is missing required columns: {', '.join(missing_columns)}")
        return dataframe

    def _read_matches(self) -> "pd.DataFrame":
        pd = self._import_pandas()
        dataframe = pd.read_parquet(self.matches_path)
        required_columns = ("qid", "dbid")
        missing_columns = [column for column in required_columns if column not in dataframe.columns]
        if missing_columns:
            raise KeyError(
                f"{self.matches_path} is missing required columns: {', '.join(missing_columns)}"
            )
        return dataframe

    def _read_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text())

    def _build_positive_index(self, matches: "pd.DataFrame") -> dict[int, list[int]]:
        positives: dict[int, list[int]] = {}
        for row in matches.itertuples(index=False):
            qid = int(row.qid)
            positives.setdefault(qid, []).append(int(row.dbid))
        return positives

    def _build_query_records(
        self, queries: "pd.DataFrame", *, database_offset: int
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in queries.itertuples(index=False):
            qid = int(row.qid)
            records.append(
                {
                    "dataset_index": database_offset + len(records),
                    "image_path": self._resolve_image_path(row.path),
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                    "positive_dbids": list(self.query_id_to_positive_dbids.get(qid, [])),
                }
            )
        return records

    def _build_database_records(self, database: "pd.DataFrame") -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for row in database.itertuples(index=False):
            records.append(
                {
                    "dataset_index": len(records),
                    "image_path": self._resolve_image_path(row.path),
                    "lat": float(row.lat),
                    "lon": float(row.lon),
                }
            )
        return records

    def _resolve_image_path(self, raw_path: str) -> str:
        image_path = Path(raw_path)
        if image_path.is_absolute():
            return str(image_path)
        if self.image_root is not None:
            return str(self.image_root / image_path)
        env_raw_dir = self._raw_dir_from_env()
        if env_raw_dir is not None:
            return str(env_raw_dir / image_path)
        return str(self.dataset_dir / image_path)

    @staticmethod
    def _import_pandas():
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VPRValidationDataset requires pandas. Install it with `pip install pandas`."
            ) from exc

        return pd

    @staticmethod
    def _raw_dir_from_env() -> Path | None:
        try:
            from datapipelines.env import raw_dir
        except ModuleNotFoundError:
            return None
        try:
            return raw_dir()
        except EnvironmentError:
            return None

    def _load_image(self, image_path: str) -> Any:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Loading images requires Pillow. Install it with `pip install pillow`."
            ) from exc

        with Image.open(image_path) as image:
            return image.convert("RGB")


def _make_validation_collate_fn():
    import torch

    def _to_tensor(image: Any) -> "torch.Tensor":
        if isinstance(image, torch.Tensor):
            return image
        try:
            from torchvision.transforms.functional import to_tensor as tvf_to_tensor

            return tvf_to_tensor(image)
        except ImportError:
            import numpy as np

            arr = np.array(image)
            if arr.ndim == 2:
                arr = arr[:, :, None]
            return torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0

    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        inputs = [_to_tensor(item["image"]) for item in batch]
        return {
            "inputs": torch.stack(inputs),
            "dataset_indices": torch.tensor([int(item["dataset_index"]) for item in batch], dtype=torch.long),
        }

    return collate_fn


def build_validation_dataloader(
    dataset_dir: str | Path,
    *,
    batch_size: int,
    image_root: str | Path | None = None,
    transform: Any = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    load_images: bool = True,
    dataset: VPRValidationDataset | None = None,
):
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Building a validation dataloader requires torch. Install it with `pip install torch`."
        ) from exc

    val_dataset = dataset or VPRValidationDataset(
        dataset_dir,
        image_root=image_root,
        transform=transform,
        load_images=load_images,
    )

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_make_validation_collate_fn(),
    )


__all__ = ["VPRValidationDataset", "build_validation_dataloader"]
