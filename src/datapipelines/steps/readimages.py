from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .base import BaseStep

IMAGE_EXTENSIONS = frozenset((".jpg", ".jpeg", ".png", ".webp"))

# ---------------------------------------------------------------------------
# WGS84 → UTM (vectorised)
# ---------------------------------------------------------------------------

_WGS84_A = 6378137.0
_WGS84_F = 1 / 298.257223563
_UTM_K0 = 0.9996
_E2 = _WGS84_F * (2 - _WGS84_F)
_EP2 = _E2 / (1 - _E2)


def _latlon_to_utm_batch(
    lat: np.ndarray, lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised WGS84 lat/lon to UTM easting/northing."""
    zone = ((lon + 180) / 6).astype(int) + 1
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)

    lat_r = np.radians(lat)
    lon_r = np.radians(lon)

    sin_lat = np.sin(lat_r)
    cos_lat = np.cos(lat_r)
    tan_lat = np.tan(lat_r)

    n = _WGS84_A / np.sqrt(1 - _E2 * sin_lat**2)
    t = tan_lat**2
    c = _EP2 * cos_lat**2
    A = cos_lat * (lon_r - lon0)

    m = _WGS84_A * (
        (1 - _E2 / 4 - 3 * _E2**2 / 64 - 5 * _E2**3 / 256) * lat_r
        - (3 * _E2 / 8 + 3 * _E2**2 / 32 + 45 * _E2**3 / 1024) * np.sin(2 * lat_r)
        + (15 * _E2**2 / 256 + 45 * _E2**3 / 1024) * np.sin(4 * lat_r)
        - (35 * _E2**3 / 3072) * np.sin(6 * lat_r)
    )

    easting = _UTM_K0 * n * (
        A + (1 - t + c) * A**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * _EP2) * A**5 / 120
    ) + 500_000.0

    northing = _UTM_K0 * (
        m + n * tan_lat * (
            A**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * A**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * _EP2) * A**6 / 720
        )
    )
    northing = np.where(lat < 0, northing + 10_000_000.0, northing)

    return easting, northing


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------


def _parse_filename(filename: str) -> tuple[float | None, float | None, float | None]:
    """Extract (utm_east, utm_north, heading) from an @-delimited filename."""
    parts = filename.rsplit(".", 1)[0].split("@")
    values = parts[1:] if parts and parts[0] == "" else parts

    def _float(v: str) -> float | None:
        try:
            return float(v.strip())
        except (ValueError, AttributeError):
            return None

    utm_east = _float(values[0]) if len(values) > 0 else None
    utm_north = _float(values[1]) if len(values) > 1 else None
    heading = _float(values[8]) if len(values) > 8 else None
    return utm_east, utm_north, heading


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _iter_sorted_image_paths(root: Path):
    """Walk *root* depth-first, yielding image paths in sorted order."""
    for current_root, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for filename in sorted(filenames):
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                yield Path(current_root) / filename


def _relative_path(path: Path, *bases: Path) -> str:
    """Return *path* relative to the first base that works."""
    for base in bases:
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def _merge_into_context(
    context: dict[str, Any], key: str, df: pd.DataFrame,
) -> dict[str, Any]:
    """Append *df* to context[key] if it already exists, re-indexing image_id."""
    existing = context.get(key)
    if existing is not None:
        df = pd.concat([existing, df], ignore_index=True)
        df["image_id"] = np.arange(len(df))
    return {**context, key: df}


def _build_cache_path(subdir: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    cache_dir = (
        Path(os.environ["PLACEFORGE_FEATURE_STORE_DIR"]) / "cache" / subdir
    )
    return cache_dir / f"{digest}.parquet"


# ---------------------------------------------------------------------------
# GSV-Cities helpers
# ---------------------------------------------------------------------------


def _gsvcities_image_path(row: pd.Series, images_dir: Path) -> Path:
    name = (
        f"{row['city_id']}"
        f"_{int(row['place_id']):07d}"
        f"_{int(row['year']):04d}"
        f"_{int(row['month']):02d}"
        f"_{int(row['northdeg']):03d}"
        f"_{row['lat']}_{row['lon']}"
        f"_{row['panoid']}.jpg"
    )
    return images_dir / row["city_id"] / name


# ---------------------------------------------------------------------------
# Train steps
# ---------------------------------------------------------------------------


class ReadTrainImagesStep(BaseStep):
    """Build a training manifest from @-delimited image filenames."""

    def __init__(
        self,
        data_root: str | Path | list[str | Path],
        source: str | None = None,
    ) -> None:
        super().__init__()
        self.data_roots = (
            [Path(d) for d in data_root]
            if isinstance(data_root, list)
            else [Path(data_root)]
        )
        self.source = source
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])

        key = "\n".join(sorted(str(r.resolve()) for r in self.data_roots))
        if self.source is not None:
            key += f"\nsource={self.source}"
        self.cache_path = _build_cache_path("readimages", key)

    def cache_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "data_roots": sorted(str(r.resolve()) for r in self.data_roots),
        }
        if self.source is not None:
            params["source"] = self.source
        return params

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
            return _merge_into_context(context, "traindataset", df)

        for root in self.data_roots:
            if not root.exists():
                raise FileNotFoundError(f"Data root does not exist: {root}")

        paths = list(self._iter_image_paths())
        if self.pbar is not None:
            self.pbar.reset(total=len(paths))

        records = []
        for image_id, image_path in enumerate(paths):
            utm_east, utm_north, heading = _parse_filename(image_path.name)
            record = {
                "image_id": image_id,
                "image_path": _relative_path(image_path, self.raw_dir),
                "utm_east": utm_east,
                "utm_north": utm_north,
                "heading": heading,
            }
            if self.source is not None:
                record["source"] = self.source
            records.append(record)
            if self.pbar is not None:
                self.pbar.update(1)

        df = pd.DataFrame(records)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.cache_path, index=False)
        return _merge_into_context(context, "traindataset", df)

    def _iter_image_paths(self):
        for root in self.data_roots:
            yield from _iter_sorted_image_paths(root)


class ReadGSVCitiesTrainImagesStep(BaseStep):
    """Build a training manifest from GSV-Cities CSVs + images."""

    def __init__(
        self,
        data_root: str | Path,
        source: str = "gsvcities",
        cities: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.source = source
        self.cities = tuple(sorted({city.strip() for city in cities if city.strip()})) if cities else None
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])

        key = str(self.data_root.resolve())
        if self.source is not None:
            key += f"\nsource={self.source}"
        if self.cities is not None:
            key += f"\ncities={','.join(self.cities)}"
        self.cache_path = _build_cache_path("readgsvcities", key)

    def cache_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"data_root": str(self.data_root.resolve())}
        if self.source is not None:
            params["source"] = self.source
        if self.cities is not None:
            params["cities"] = list(self.cities)
        return params

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.cache_path.exists():
            df = pd.read_parquet(self.cache_path)
            return _merge_into_context(context, "traindataset", df)

        dataframes_dir = self.data_root / "Dataframes"
        images_dir = self.data_root / "Images"

        if not dataframes_dir.exists():
            raise FileNotFoundError(
                f"Dataframes dir does not exist: {dataframes_dir}"
            )

        csv_paths = sorted(dataframes_dir.glob("*.csv"))
        if self.cities is not None:
            csv_paths_by_stem = {path.stem: path for path in csv_paths}
            missing_cities = [
                city for city in self.cities if city not in csv_paths_by_stem
            ]
            if missing_cities:
                raise ValueError(
                    "Requested GSV-Cities dataframes were not found: "
                    f"{missing_cities}. Available cities: "
                    f"{sorted(csv_paths_by_stem)}"
                )
            csv_paths = [csv_paths_by_stem[city] for city in self.cities]

        df = pd.concat(
            [pd.read_csv(p) for p in csv_paths], ignore_index=True,
        )

        if self.pbar is not None:
            self.pbar.reset(total=len(df))

        image_paths: list[str] = []
        for _, row in df.iterrows():
            path = _gsvcities_image_path(row, images_dir)
            image_paths.append(
                _relative_path(path, self.raw_dir, self.data_root)
            )
            if self.pbar is not None:
                self.pbar.update(1)

        easting, northing = _latlon_to_utm_batch(
            df["lat"].to_numpy(dtype=np.float64),
            df["lon"].to_numpy(dtype=np.float64),
        )

        result = pd.DataFrame({
            "image_id": np.arange(len(df)),
            "image_path": image_paths,
            "place_id": df["place_id"].to_numpy(dtype=np.int64),
            "utm_east": easting,
            "utm_north": northing,
            "heading": df["northdeg"].to_numpy(dtype=np.float64),
        })

        if self.source is not None:
            result["source"] = self.source

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(self.cache_path, index=False)
        return _merge_into_context(context, "traindataset", result)


# ---------------------------------------------------------------------------
# Eval steps (val / test)
# ---------------------------------------------------------------------------


class _ReadEvalImagesStep(BaseStep):
    """Shared logic for validation and test image reading."""

    context_key: str  # set by subclasses

    def __init__(self, query_path: str | Path, database_path: str | Path) -> None:
        super().__init__()
        self.raw_dir = Path(os.environ["PLACEFORGE_RAW_DIR"])
        self.query_path = self.raw_dir / query_path
        self.database_path = self.raw_dir / database_path

    def cache_params(self) -> dict[str, Any]:
        return {
            "query_path": str(self.query_path),
            "database_path": str(self.database_path),
        }

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        for path in (self.query_path, self.database_path):
            if not path.exists():
                raise FileNotFoundError(f"Directory does not exist: {path}")

        query_paths = list(_iter_sorted_image_paths(self.query_path))
        database_paths = list(_iter_sorted_image_paths(self.database_path))

        all_paths = [(p, True) for p in query_paths] + [
            (p, False) for p in database_paths
        ]

        if self.pbar is not None:
            self.pbar.reset(total=len(all_paths))

        records = []
        for image_id, (image_path, is_query) in enumerate(all_paths):
            utm_east, utm_north, _ = _parse_filename(image_path.name)
            records.append({
                "image_id": image_id,
                "image_path": _relative_path(image_path, self.raw_dir),
                "is_query": is_query,
                "utm_east": utm_east,
                "utm_north": utm_north,
            })
            if self.pbar is not None:
                self.pbar.update(1)

        return {**context, self.context_key: pd.DataFrame(records)}


class ReadValImagesStep(_ReadEvalImagesStep):
    context_key = "valdataset"


class ReadTestImagesStep(_ReadEvalImagesStep):
    context_key = "testdataset"
