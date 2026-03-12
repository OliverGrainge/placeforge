from __future__ import annotations

from math import floor
from typing import Any

from .base import BaseStep


class AssignSuperGroupsStep(BaseStep):
    """Assigns a supergroup_id to each image such that all place_ids within
    the same supergroup are guaranteed non-adjacent.

    Uses a repeating ``(adjacency_cells + 1) x (adjacency_cells + 1)`` grid
    coloring.  With ``adjacency_cells=1`` (Moore neighbourhood) this produces
    4 supergroups in a 2×2 repeating pattern::

        0 1 0 1 …
        2 3 2 3 …
        0 1 0 1 …
        2 3 2 3 …

    Same-supergroup cells have Chebyshev distance ≥ ``adjacency_cells + 1``,
    so they are never grid neighbours.  In general, ``(adjacency_cells + 1)²``
    supergroups are produced.

    Training guarantee: any two *distinct* place_ids drawn from the same
    supergroup are spatially separated — their images cannot be spatial matches
    of each other, making same-supergroup pairs safe to use as hard negatives.

    Parameters
    ----------
    cell_size_meters:
        Must match the value used in ``AssignPlaceIdsStep`` so the grids align.
    adjacency_cells:
        Chebyshev radius (in grid cells) that defines "adjacent".  1 (default)
        = Moore neighbourhood (8 neighbours).
    """

    def __init__(
        self,
        cell_size_meters: float,
        *,
        adjacency_cells: int = 1,
        context_key: str = "index",
        supergroup_id_column: str = "supergroup_id",
        easting_column: str = "utm_east",
        northing_column: str = "utm_north",
        zone_number_column: str = "utm_zone_number",
        zone_letter_column: str = "utm_zone_letter",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if cell_size_meters <= 0:
            raise ValueError("cell_size_meters must be positive")
        if adjacency_cells < 1:
            raise ValueError("adjacency_cells must be >= 1")
        self.cell_size_meters = cell_size_meters
        self.adjacency_cells = adjacency_cells
        self.context_key = context_key
        self.supergroup_id_column = supergroup_id_column
        self.easting_column = easting_column
        self.northing_column = northing_column
        self.zone_number_column = zone_number_column
        self.zone_letter_column = zone_letter_column

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.context_key not in context:
            raise KeyError(f"Context is missing dataframe at key {self.context_key!r}")

        dataframe = context[self.context_key]
        self._validate_columns(dataframe)

        supergroup_ids = self._assign_supergroups(dataframe)
        result = dataframe.copy()
        result[self.supergroup_id_column] = supergroup_ids

        context = dict(context)
        context[self.context_key] = result
        return context

    def _assign_supergroups(self, dataframe: Any) -> list[int | None]:
        eastings = dataframe[self.easting_column].tolist()
        northings = dataframe[self.northing_column].tolist()
        cell_size = self.cell_size_meters
        # period = k+1 guarantees same-supergroup cells are Chebyshev-distance >= k+1
        period = self.adjacency_cells + 1

        supergroup_ids: list[int | None] = []
        with self.progress(total=len(dataframe), desc="assign supergroups") as progress:
            for easting, northing in zip(eastings, northings, strict=False):
                if easting is None or northing is None:
                    supergroup_ids.append(None)
                else:
                    cell_x = floor(easting / cell_size)
                    cell_y = floor(northing / cell_size)
                    supergroup_ids.append((cell_x % period) * period + (cell_y % period))
                progress.update(1)
        return supergroup_ids

    def _validate_columns(self, dataframe: Any) -> None:
        required = (
            self.easting_column,
            self.northing_column,
            self.zone_number_column,
            self.zone_letter_column,
        )
        missing = [c for c in required if c not in dataframe.columns]
        if missing:
            raise KeyError(
                f"Dataframe at {self.context_key!r} is missing columns: {', '.join(missing)}"
            )
