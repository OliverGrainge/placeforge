from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseStep


class ReadImagesStep(BaseStep):
    def __init__(self, data_root: str | Path) -> None:
        super().__init__()
        self.data_root = Path(data_root)

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ReadImagesStep requires pandas. Install it with `pip install pandas`."
            ) from exc

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

        records = []
        for image_id, image_path in enumerate(self._iter_image_paths()):
            utm_east, utm_north = self._parse_utm(image_path.name)
            records.append({
                "image_id": image_id,
                "image_path": str(image_path.relative_to(self.data_root)),
                "utm_east": utm_east,
                "utm_north": utm_north,
            })

        return {**context, "dataset": pd.DataFrame(records)}

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