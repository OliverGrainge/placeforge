from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .base import BaseStep


class SaveTrainDataset(BaseStep):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        processed_dir = os.environ["PLACEFORGE_PROCESSED_DIR"]
        output_path = Path(processed_dir) / self.name / "dataset.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        context["dataset"].to_parquet(output_path)

        return context