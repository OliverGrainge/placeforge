from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd 
import os

from .base import BaseStep


class ReadTrainImagesStep(BaseStep):
    def __init__(self, data_root: str | Path) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        paths = list(self._iter_image_paths())
        if self.pbar is not None:
            self.pbar.reset(total=len(paths))

        records = []
        for image_id, image_path in enumerate(paths):
            utm_east, utm_north = self._parse_utm(image_path.name)
            records.append({
                "image_id": image_id,
                "image_path": str(image_path.relative_to(self.raw_dir)),
                "utm_east": utm_east,
                "utm_north": utm_north,
            })
            if self.pbar is not None:
                self.pbar.update(1)

        return {**context, "traindataset": pd.DataFrame(records)}

    def _iter_image_paths(self):
        for path in sorted(self.data_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                yield path

    def _parse_utm(self, filename: str) -> tuple[float | None, float | None]:
        parts = filename.rsplit(".", 1)[0].split("@")
        values = parts[1:] if parts and parts[0] == "" else parts

        def to_float(v: str) -> float | None:
            try:
                return float(v.strip())
            except (ValueError, AttributeError):
                return None

        utm_east = to_float(values[0]) if len(values) > 0 else None
        utm_north = to_float(values[1]) if len(values) > 1 else None
        return utm_east, utm_north



class ReadValImagesStep(BaseStep):
    def __init__(self, query_path: str | Path, database_path: str | Path) -> None:
        super().__init__()
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
        self.query_path = self.raw_dir / query_path
        self.database_path = self.raw_dir / database_path

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        for path in (self.query_path, self.database_path):
            if not path.exists():
                raise FileNotFoundError(f"Directory does not exist: {path}")

        query_paths = list(self._iter_image_paths(self.query_path))
        database_paths = list(self._iter_image_paths(self.database_path))

        if self.pbar is not None:
            self.pbar.reset(total=len(query_paths) + len(database_paths))

        records = []
        for image_id, (image_path, is_query) in enumerate(
            [(p, True) for p in query_paths] + [(p, False) for p in database_paths]
        ):
            records.append({
                "image_id": image_id,
                "image_path": str(image_path.relative_to(self.raw_dir)),
                "is_query": is_query,
            })
            if self.pbar is not None:
                self.pbar.update(1)

        return {**context, "valdataset": pd.DataFrame(records)}

    def _iter_image_paths(self, root: Path):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                yield path