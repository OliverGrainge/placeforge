from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import Any

from .base import BaseStep


class AddRadiusMatchesStep(BaseStep):
    def __init__(
        self,
        radius_meters: float,
        *,
        context_key: str = "index",
        matches_column: str = "matches",
        image_id_column: str = "image_id",
        easting_column: str = "utm_east",
        northing_column: str = "utm_north",
        zone_number_column: str = "utm_zone_number",
        zone_letter_column: str = "utm_zone_letter",
        include_self: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if radius_meters <= 0:
            raise ValueError("radius_meters must be greater than 0")

        self.radius_meters = radius_meters
        self.context_key = context_key
        self.matches_column = matches_column
        self.image_id_column = image_id_column
        self.easting_column = easting_column
        self.northing_column = northing_column
        self.zone_number_column = zone_number_column
        self.zone_letter_column = zone_letter_column
        self.include_self = include_self

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.context_key not in context:
            raise KeyError(f"Context is missing dataframe at key {self.context_key!r}")

        dataframe = context[self.context_key]
        self._validate_columns(dataframe)

        matches = self._build_matches(dataframe)
        result = dataframe.copy()
        result[self.matches_column] = matches

        context = dict(context)
        context[self.context_key] = result
        return context

    def _build_matches(self, dataframe: Any) -> list[list[Any]]:
        image_ids = dataframe[self.image_id_column].tolist()
        eastings = dataframe[self.easting_column].tolist()
        northings = dataframe[self.northing_column].tolist()
        zone_numbers = dataframe[self.zone_number_column].tolist()
        zone_letters = dataframe[self.zone_letter_column].tolist()

        radius_squared = self.radius_meters * self.radius_meters
        cell_size = self.radius_meters
        spatial_index: dict[tuple[Any, Any, int, int], list[tuple[int, float, float]]] = defaultdict(list)
        matches: list[list[Any]] = [[] for _ in range(len(dataframe))]

        with self.progress(total=len(dataframe), desc="radius matches") as progress:
            for idx, image_id in enumerate(image_ids):
                easting = eastings[idx]
                northing = northings[idx]
                zone_number = zone_numbers[idx]
                zone_letter = zone_letters[idx]

                if easting is None or northing is None:
                    progress.update(1)
                    continue

                zone_key = (zone_number, zone_letter)
                cell_x = floor(easting / cell_size)
                cell_y = floor(northing / cell_size)

                for neighbor_x in range(cell_x - 1, cell_x + 2):
                    for neighbor_y in range(cell_y - 1, cell_y + 2):
                        cell_key = (*zone_key, neighbor_x, neighbor_y)
                        for other_idx, other_easting, other_northing in spatial_index.get(cell_key, []):
                            delta_east = easting - other_easting
                            delta_north = northing - other_northing
                            if delta_east * delta_east + delta_north * delta_north > radius_squared:
                                continue

                            other_image_id = image_ids[other_idx]
                            matches[idx].append(other_image_id)
                            matches[other_idx].append(image_id)

                if self.include_self:
                    matches[idx].append(image_id)

                spatial_index[(*zone_key, cell_x, cell_y)].append((idx, easting, northing))
                progress.update(1)

        return matches

    def _validate_columns(self, dataframe: Any) -> None:
        required_columns = (
            self.image_id_column,
            self.easting_column,
            self.northing_column,
            self.zone_number_column,
            self.zone_letter_column,
        )
        missing_columns = [column for column in required_columns if column not in dataframe.columns]
        if missing_columns:
            missing = ", ".join(missing_columns)
            raise KeyError(f"Dataframe at {self.context_key!r} is missing columns: {missing}")
