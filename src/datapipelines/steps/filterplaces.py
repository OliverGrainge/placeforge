from __future__ import annotations

from typing import Any

from .base import BaseStep


class RemoveSmallPlacesStep(BaseStep):
    def __init__(
        self,
        min_images_per_place: int | None = None,
        *,
        min_images_per_place_context_key: str | None = None,
        context_key: str = "index",
        place_id_column: str = "place_id",
        image_id_column: str = "image_id",
        matches_column: str = "matches",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if min_images_per_place is not None and min_images_per_place < 1:
            raise ValueError("min_images_per_place must be >= 1")
        self.min_images_per_place = min_images_per_place
        self.min_images_per_place_context_key = min_images_per_place_context_key
        self.context_key = context_key
        self.place_id_column = place_id_column
        self.image_id_column = image_id_column
        self.matches_column = matches_column

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.context_key not in context:
            raise KeyError(f"Context is missing dataframe at key {self.context_key!r}")
        with self.progress(total=1, desc="remove small places") as progress:
            dataframe = context[self.context_key]
            if self.place_id_column not in dataframe.columns:
                raise KeyError(
                    f"Dataframe at {self.context_key!r} is missing place_id column {self.place_id_column!r}"
                )

            min_images_per_place = self._resolve_min_images_per_place(context)
            if min_images_per_place is None or min_images_per_place <= 1:
                progress.update(1)
                return context

            place_counts = dataframe[self.place_id_column].value_counts(dropna=True)
            valid_place_ids = set(place_counts[place_counts >= min_images_per_place].index.tolist())
            mask = dataframe[self.place_id_column].isin(valid_place_ids)
            filtered = dataframe.loc[mask].reset_index(drop=True)
            filtered = self._prune_matches(filtered)

            context = dict(context)
            context[self.context_key] = filtered
            progress.update(1)
            return context

    def _resolve_min_images_per_place(self, context: dict[str, Any]) -> int | None:
        if self.min_images_per_place_context_key is not None:
            value = context.get(self.min_images_per_place_context_key, self.min_images_per_place)
        else:
            value = self.min_images_per_place

        if value is None:
            return None
        if not isinstance(value, int):
            raise TypeError("min_images_per_place must be an int")
        if value < 1:
            raise ValueError("min_images_per_place must be >= 1")
        return value

    def _prune_matches(self, dataframe: Any) -> Any:
        if self.matches_column not in dataframe.columns or self.image_id_column not in dataframe.columns:
            return dataframe

        valid_image_ids = {str(image_id) for image_id in dataframe[self.image_id_column].tolist()}
        result = dataframe.copy()
        result[self.matches_column] = result[self.matches_column].apply(
            lambda matches: self._filter_match_list(matches, valid_image_ids)
        )
        return result

    @staticmethod
    def _filter_match_list(matches: Any, valid_image_ids: set[str]) -> list[Any]:
        if not isinstance(matches, list):
            return []

        filtered: list[Any] = []
        for image_id in matches:
            if str(image_id) in valid_image_ids:
                filtered.append(image_id)
        return filtered
