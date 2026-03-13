from __future__ import annotations

from math import floor
from typing import Any

from .base import BaseStep


class AssignPlaceIdStep(BaseStep):
    def __init__(self, cell_size_meters: float) -> None:
        super().__init__()
        self.cell_size_meters = cell_size_meters

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["dataset"].copy()
        cell_size = self.cell_size_meters

        df["cell_x"] = (df["utm_east"] / cell_size).apply(floor)
        df["cell_y"] = (df["utm_north"] / cell_size).apply(floor)
        df["place_id"] = df["cell_x"] * df["cell_y"].max() + df["cell_y"]

        return {**context, "dataset": df}