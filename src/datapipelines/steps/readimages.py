from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import os

from .base import BaseStep


def _parse_filename(filename: str) -> tuple[float | None, float | None, float | None]:
    parts = filename.rsplit(".", 1)[0].split("@")
    values = parts[1:] if parts and parts[0] == "" else parts

    def to_float(v: str) -> float | None:
        try:
            return float(v.strip())
        except (ValueError, AttributeError):
            return None

    utm_east = to_float(values[0]) if len(values) > 0 else None
    utm_north = to_float(values[1]) if len(values) > 1 else None
    heading = to_float(values[8]) if len(values) > 8 else None
    return utm_east, utm_north, heading


def _parse_utm(filename: str) -> tuple[float | None, float | None]:
    utm_east, utm_north, _ = _parse_filename(filename)
    return utm_east, utm_north


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def _iter_sorted_image_paths(root: Path):
    for current_root, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                yield Path(current_root) / filename


class ReadTrainImagesStep(BaseStep):
    def __init__(
        self,
        data_root: str | Path | list[str | Path],
    ) -> None:
        super().__init__()
        if isinstance(data_root, list):
            self.data_roots = [Path(d) for d in data_root]
        else:
            self.data_roots = [Path(data_root)]
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        for data_root in self.data_roots:
            if not data_root.exists():
                raise FileNotFoundError(f"Data root does not exist: {data_root}")

        paths = list(self._iter_image_paths())
        if self.pbar is not None:
            self.pbar.reset(total=len(paths))

        records = []
        for image_id, image_path in enumerate(paths):
            utm_east, utm_north, heading = _parse_filename(image_path.name)
            records.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path.relative_to(self.raw_dir)),
                    "utm_east": utm_east,
                    "utm_north": utm_north,
                    "heading": heading,
                }
            )
            if self.pbar is not None:
                self.pbar.update(1)
        return {**context, "traindataset": pd.DataFrame(records)}

    def _iter_image_paths(self):
        for data_root in self.data_roots:
            yield from _iter_sorted_image_paths(data_root)


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
            utm_east, utm_north = _parse_utm(image_path.name)
            records.append(
                {
                    "image_id": image_id,
                    "image_path": str(image_path.relative_to(self.raw_dir)),
                    "is_query": is_query,
                    "utm_east": utm_east,
                    "utm_north": utm_north,
                }
            )
            if self.pbar is not None:
                self.pbar.update(1)

        return {**context, "valdataset": pd.DataFrame(records)}

    def _iter_image_paths(self, root: Path):
        yield from _iter_sorted_image_paths(root)
