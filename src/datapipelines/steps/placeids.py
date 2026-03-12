from __future__ import annotations

from math import floor
from typing import Any

from .base import BaseStep


class AssignPlaceIdsStep(BaseStep):
    """Assigns each image a place_id based on a spatial grid.

    Each occupied grid cell becomes a distinct place_id. Images in the same
    cell share a place_id and form the positive set for metric learning.
    Images in different cells have non-overlapping image sets, making them
    safe to treat as separate classes.

    The place_id string encodes the UTM zone and cell coordinates:
    ``"{zone_number}_{zone_letter}_{cell_x}_{cell_y}"``
    """

    def __init__(
        self,
        cell_size_meters: float,
        *,
        context_key: str = "index",
        place_id_column: str = "place_id",
        easting_column: str = "utm_east",
        northing_column: str = "utm_north",
        zone_number_column: str = "utm_zone_number",
        zone_letter_column: str = "utm_zone_letter",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if cell_size_meters <= 0:
            raise ValueError("cell_size_meters must be positive")
        self.cell_size_meters = cell_size_meters
        self.context_key = context_key
        self.place_id_column = place_id_column
        self.easting_column = easting_column
        self.northing_column = northing_column
        self.zone_number_column = zone_number_column
        self.zone_letter_column = zone_letter_column

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.context_key not in context:
            raise KeyError(f"Context is missing dataframe at key {self.context_key!r}")

        dataframe = context[self.context_key]
        self._validate_columns(dataframe)

        place_ids = self._assign_place_ids(dataframe)
        result = dataframe.copy()
        result[self.place_id_column] = place_ids

        context = dict(context)
        context[self.context_key] = result
        return context

    def _assign_place_ids(self, dataframe: Any) -> list[str | None]:
        eastings = dataframe[self.easting_column].tolist()
        northings = dataframe[self.northing_column].tolist()
        zone_numbers = dataframe[self.zone_number_column].tolist()
        zone_letters = dataframe[self.zone_letter_column].tolist()
        cell_size = self.cell_size_meters

        place_ids: list[str | None] = []
        with self.progress(total=len(dataframe), desc="assign place ids") as progress:
            for easting, northing, zone_number, zone_letter in zip(
                eastings, northings, zone_numbers, zone_letters, strict=False
            ):
                if easting is None or northing is None:
                    place_ids.append(None)
                else:
                    cell_x = floor(easting / cell_size)
                    cell_y = floor(northing / cell_size)
                    place_ids.append(f"{zone_number}_{zone_letter}_{cell_x}_{cell_y}")
                progress.update(1)
        return place_ids

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
