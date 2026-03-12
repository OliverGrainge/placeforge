from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .base import BaseStep

if TYPE_CHECKING:
    import pandas as pd

_FILENAME_COLUMNS = (
    "utm_east",
    "utm_north",
    "utm_zone_number",
    "utm_zone_letter",
    "latitude",
    "longitude",
    "pano_id",
    "_unused_1",
    "heading",
    "_unused_2",
    "_unused_3",
    "_unused_4",
    "timestamp",
    "_unused_5",
    "_unused_6",
)


class ReadImagesStep(BaseStep):
    def __init__(
        self,
        data_root: str | Path,
        *,
        metadata_filename: str = "metadata.csv",
        context_key: str = "index",
        recursive: bool = True,
        image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.data_root = Path(data_root)
        self.metadata_filename = metadata_filename
        self.context_key = context_key
        self.recursive = recursive
        self.image_extensions = tuple(ext.lower() for ext in image_extensions)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        pd = self._import_pandas()

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        records = {column: [] for column in self._record_columns()}

        with self.progress(desc="scan images") as progress:
            for image_path in self._iter_image_paths():
                record = self._build_record(image_path)
                for key, value in record.items():
                    records[key].append(value)
                progress.update(1)

        dataframe = pd.DataFrame.from_dict(records)
        metadata = self._read_metadata_csv()

        if metadata is not None:
            dataframe = self._merge_metadata(dataframe, metadata)

        context = dict(context)
        context[self.context_key] = dataframe
        return context

    def _iter_image_paths(self):
        iterator = self.data_root.rglob("*") if self.recursive else self.data_root.glob("*")

        for path in iterator:
            if path.is_file() and path.suffix.lower() in self.image_extensions:
                yield path

    def _record_columns(self) -> tuple[str, ...]:
        return ("image_id", "image_path", "filename", *self._metadata_columns())

    def _metadata_columns(self) -> tuple[str, ...]:
        return tuple(column for column in _FILENAME_COLUMNS if not column.startswith("_unused_"))

    def _build_record(self, image_path: Path) -> dict[str, Any]:
        parsed_metadata = self._parse_image_name(image_path.name)
        return {
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "filename": image_path.name,
            **parsed_metadata,
        }

    def _parse_image_name(self, filename: str) -> dict[str, Any]:
        stem = filename.rsplit(".", 1)[0]
        parts = stem.split("@")
        values = parts[1:] if parts and parts[0] == "" else parts
        values = values[: len(_FILENAME_COLUMNS)] + [""] * max(0, len(_FILENAME_COLUMNS) - len(values))

        parsed = {
            column: self._clean_value(raw_value)
            for column, raw_value in zip(_FILENAME_COLUMNS, values, strict=False)
            if not column.startswith("_unused_")
        }

        for numeric_column in ("utm_east", "utm_north", "latitude", "longitude", "heading"):
            parsed[numeric_column] = self._to_float(parsed[numeric_column])

        parsed["utm_zone_number"] = self._to_int(parsed["utm_zone_number"])
        return parsed

    def _read_metadata_csv(self) -> "pd.DataFrame | None":
        pd = self._import_pandas()
        metadata_path = self.data_root / self.metadata_filename
        if not metadata_path.exists():
            return None

        return pd.read_csv(metadata_path, low_memory=False)

    def _merge_metadata(self, images: "pd.DataFrame", metadata: "pd.DataFrame") -> "pd.DataFrame":
        if metadata.empty:
            return images

        join_key = self._resolve_join_key(images.columns, metadata.columns)
        if join_key is None:
            return images

        merged = images.merge(metadata, how="left", on=join_key, suffixes=("", "_metadata"))
        duplicate_columns = [
            column for column in merged.columns if column.endswith("_metadata") and column[:-9] in images.columns
        ]
        return merged.drop(columns=duplicate_columns)

    @staticmethod
    def _import_pandas():
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ReadImagesStep requires pandas. Install it with `pip install pandas`."
            ) from exc

        return pd

    def _resolve_join_key(self, image_columns: Any, metadata_columns: Any) -> str | None:
        candidates = ("image_id", "pano_id", "panoid", "panorama_id", "id")
        metadata_set = set(metadata_columns)
        image_set = set(image_columns)

        for candidate in candidates:
            if candidate in image_set and candidate in metadata_set:
                return candidate

        return None

    @staticmethod
    def _clean_value(value: str) -> str | None:
        value = value.strip()
        return value or None

    @staticmethod
    def _to_float(value: str | None) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except ValueError:
            return None

    @staticmethod
    def _to_int(value: str | None) -> int | None:
        if value is None:
            return None

        try:
            return int(value)
        except ValueError:
            return None
