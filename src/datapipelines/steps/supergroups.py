from __future__ import annotations

from typing import Any

from .base import BaseStep


class AssignSuperGroupStep(BaseStep):
    def __init__(self, adjacency_cells: int = 1) -> None:
        super().__init__()
        self.adjacency_cells = adjacency_cells

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        df = context["dataset"].copy()
        period = self.adjacency_cells + 1

        df["supergroup_id"] = (df["cell_x"] % period) * period + (df["cell_y"] % period)

        return {**context, "dataset": df}