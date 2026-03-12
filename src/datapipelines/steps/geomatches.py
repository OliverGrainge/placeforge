from __future__ import annotations

from collections import defaultdict
from math import floor
from typing import Any

from .base import BaseStep


class ComputeGeoMatchesStep(BaseStep):
    """Cross-match query images against database images by geographic proximity.

    Reads the query and database DataFrames from context (both must already
    have integer ID columns assigned, e.g. by :class:`AssignSplitIdsStep`),
    builds a UTM-based spatial grid index over the database images, and finds
    every database image within *radius_meters* of each query image.

    The resulting ``matches`` DataFrame (columns ``qid``, ``dbid``, sorted by
    ``qid``) is written to *matches_context_key* in context.
    """

    def __init__(
        self,
        radius_meters: float,
        *,
        queries_context_key: str = "queries",
        database_context_key: str = "database",
        matches_context_key: str = "matches",
        query_id_column: str = "qid",
        database_id_column: str = "dbid",
        easting_column: str = "utm_east",
        northing_column: str = "utm_north",
        zone_number_column: str = "utm_zone_number",
        zone_letter_column: str = "utm_zone_letter",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if radius_meters <= 0:
            raise ValueError("radius_meters must be greater than 0")

        self.radius_meters = radius_meters
        self.queries_context_key = queries_context_key
        self.database_context_key = database_context_key
        self.matches_context_key = matches_context_key
        self.query_id_column = query_id_column
        self.database_id_column = database_id_column
        self.easting_column = easting_column
        self.northing_column = northing_column
        self.zone_number_column = zone_number_column
        self.zone_letter_column = zone_letter_column

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ComputeGeoMatchesStep requires pandas. Install it with `pip install pandas`."
            ) from exc

        for key in (self.queries_context_key, self.database_context_key):
            if key not in context:
                raise KeyError(f"Context is missing dataframe at key {key!r}")

        queries = context[self.queries_context_key]
        database = context[self.database_context_key]

        self._validate_columns(queries, "queries", extra=(self.query_id_column,))
        self._validate_columns(database, "database", extra=(self.database_id_column,))

        match_pairs = self._build_matches(queries, database)

        if match_pairs:
            qids, dbids = zip(*match_pairs)
            matches_df = pd.DataFrame(
                {
                    "qid": pd.array(list(qids), dtype="uint32"),
                    "dbid": pd.array(list(dbids), dtype="uint32"),
                }
            )
        else:
            matches_df = pd.DataFrame(
                {"qid": pd.array([], dtype="uint32"), "dbid": pd.array([], dtype="uint32")}
            )

        matches_df = matches_df.sort_values("qid").reset_index(drop=True)

        context = dict(context)
        context[self.matches_context_key] = matches_df
        return context

    def _build_matches(self, queries: Any, database: Any) -> list[tuple[int, int]]:
        db_eastings = database[self.easting_column].tolist()
        db_northings = database[self.northing_column].tolist()
        db_zone_numbers = database[self.zone_number_column].tolist()
        db_zone_letters = database[self.zone_letter_column].tolist()
        db_ids = database[self.database_id_column].tolist()

        q_eastings = queries[self.easting_column].tolist()
        q_northings = queries[self.northing_column].tolist()
        q_zone_numbers = queries[self.zone_number_column].tolist()
        q_zone_letters = queries[self.zone_letter_column].tolist()
        q_ids = queries[self.query_id_column].tolist()

        radius_squared = self.radius_meters * self.radius_meters
        cell_size = self.radius_meters

        # Build a spatial grid index over database images.
        spatial_index: dict[tuple[Any, Any, int, int], list[tuple[int, float, float]]] = defaultdict(list)
        for db_idx in range(len(database)):
            easting = db_eastings[db_idx]
            northing = db_northings[db_idx]
            if easting is None or northing is None:
                continue
            cell_x = floor(easting / cell_size)
            cell_y = floor(northing / cell_size)
            spatial_index[
                (db_zone_numbers[db_idx], db_zone_letters[db_idx], cell_x, cell_y)
            ].append((db_idx, easting, northing))

        matches: list[tuple[int, int]] = []

        with self.progress(total=len(queries), desc="geo matches") as progress:
            for q_idx in range(len(queries)):
                q_easting = q_eastings[q_idx]
                q_northing = q_northings[q_idx]
                if q_easting is None or q_northing is None:
                    progress.update(1)
                    continue

                cell_x = floor(q_easting / cell_size)
                cell_y = floor(q_northing / cell_size)
                q_zone_num = q_zone_numbers[q_idx]
                q_zone_let = q_zone_letters[q_idx]
                qid = int(q_ids[q_idx])

                for neighbor_x in range(cell_x - 1, cell_x + 2):
                    for neighbor_y in range(cell_y - 1, cell_y + 2):
                        cell_key = (q_zone_num, q_zone_let, neighbor_x, neighbor_y)
                        for db_idx, db_easting, db_northing in spatial_index.get(cell_key, []):
                            delta_east = q_easting - db_easting
                            delta_north = q_northing - db_northing
                            if delta_east * delta_east + delta_north * delta_north <= radius_squared:
                                matches.append((qid, int(db_ids[db_idx])))

                progress.update(1)

        return matches

    def _validate_columns(self, dataframe: Any, label: str, extra: tuple[str, ...] = ()) -> None:
        required = (
            self.easting_column,
            self.northing_column,
            self.zone_number_column,
            self.zone_letter_column,
            *extra,
        )
        missing = [col for col in required if col not in dataframe.columns]
        if missing:
            raise KeyError(f"{label} dataframe is missing columns: {', '.join(missing)}")
