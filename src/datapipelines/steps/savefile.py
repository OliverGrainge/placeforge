from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseStep


class SaveFileStep(BaseStep):
    def __init__(
        self,
        output_path: str | Path,
        *,
        context_key: str = "index",
        output_path_context_key: str = "saved_file_path",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.output_path = Path(output_path)
        self.context_key = context_key
        self.output_path_context_key = output_path_context_key

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.context_key not in context:
            raise KeyError(f"Context is missing value at key {self.context_key!r}")
        with self.progress(total=1, desc="save file") as progress:
            value = context[self.context_key]
            self._write_value(value)

            context = dict(context)
            context[self.output_path_context_key] = str(self.output_path)
            progress.update(1)
            return context

    def _write_value(self, value: Any) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(value, "to_parquet"):
            self._write_parquet(value)
            return

        if isinstance(value, dict):
            self.output_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
            return

        raise TypeError(
            f"Context value at {self.context_key!r} does not support persistence via SaveFileStep"
        )

    def _write_parquet(self, dataframe: Any) -> None:
        try:
            dataframe.to_parquet(self.output_path, index=False)
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Writing parquet requires `pyarrow` or `fastparquet`. "
                "Install one of them, for example `pip install pyarrow`."
            ) from exc
