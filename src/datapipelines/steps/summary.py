from __future__ import annotations

from typing import Any

from .base import BaseStep


class DatasetSummaryStep(BaseStep):
    def __init__(
        self,
        *,
        context_key: str = "index",
        output_context_key: str = "stats",
        place_id_column: str = "place_id",
        supergroup_id_column: str = "supergroup_id",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.context_key = context_key
        self.output_context_key = output_context_key
        self.place_id_column = place_id_column
        self.supergroup_id_column = supergroup_id_column

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.context_key not in context:
            raise KeyError(f"Context is missing dataframe at key {self.context_key!r}")

        dataframe = context[self.context_key]
        stats = self._build_summary(dataframe)

        context = dict(context)
        context[self.output_context_key] = stats
        return context

    def _build_summary(self, dataframe: Any) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "total_images": len(dataframe),
        }

        place_counts = self._counts_for_column(dataframe, self.place_id_column)
        stats["num_places"] = len(place_counts)
        stats.update(self._summarize_counts(place_counts, "images_per_place"))

        supergroup_counts = self._counts_for_column(dataframe, self.supergroup_id_column)
        stats["num_supergroups"] = len(supergroup_counts)
        stats.update(self._summarize_counts(supergroup_counts, "images_per_supergroup"))

        supergroup_place_counts = self._supergroup_place_counts_from_dataframe(dataframe)
        stats.update(self._summarize_counts(supergroup_place_counts, "places_per_supergroup"))

        return stats

    def _counts_for_column(self, dataframe: Any, column: str) -> list[int]:
        if column not in dataframe.columns:
            return []
        return self._group_counts(dataframe[column])

    @staticmethod
    def _group_counts(series: Any) -> list[int]:
        counts: dict[Any, int] = {}
        for value in series:
            if value is not None:
                counts[value] = counts.get(value, 0) + 1
        return list(counts.values())

    @staticmethod
    def _supergroup_place_counts(place_series: Any, supergroup_series: Any) -> list[int]:
        seen: dict[Any, set] = {}
        for place_id, sg_id in zip(place_series, supergroup_series, strict=False):
            if sg_id is not None and place_id is not None:
                if sg_id not in seen:
                    seen[sg_id] = set()
                seen[sg_id].add(place_id)
        return [len(places) for places in seen.values()]

    def _supergroup_place_counts_from_dataframe(self, dataframe: Any) -> list[int]:
        if self.supergroup_id_column not in dataframe.columns or self.place_id_column not in dataframe.columns:
            return []
        return self._supergroup_place_counts(
            dataframe[self.place_id_column],
            dataframe[self.supergroup_id_column],
        )

    def _summarize_counts(self, values: list[int], label: str) -> dict[str, Any]:
        if not values:
            return {
                f"mean_{label}": 0.0,
                f"median_{label}": 0.0,
                f"min_{label}": 0,
                f"max_{label}": 0,
            }

        return {
            f"mean_{label}": sum(values) / len(values),
            f"median_{label}": self._median(values),
            f"min_{label}": min(values),
            f"max_{label}": max(values),
        }

    @staticmethod
    def _median(values: list[int]) -> float:
        if not values:
            return 0.0

        ordered = sorted(values)
        midpoint = len(ordered) // 2
        if len(ordered) % 2 == 1:
            return float(ordered[midpoint])
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
