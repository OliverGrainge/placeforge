from __future__ import annotations

from typing import Any

from .base import BaseStep


class RemoveUnmatchedImagesStep(BaseStep):
    def __init__(
        self,
        *,
        context_key: str = "index",
        matches_column: str = "matches",
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.context_key = context_key
        self.matches_column = matches_column

    def run(self, context: dict[str, Any]) -> dict[str, Any]:
        if self.context_key not in context:
            raise KeyError(f"Context is missing dataframe at key {self.context_key!r}")
        with self.progress(total=1, desc="remove unmatched images") as progress:
            dataframe = context[self.context_key]
            if self.matches_column not in dataframe.columns:
                raise KeyError(
                    f"Dataframe at {self.context_key!r} is missing matches column {self.matches_column!r}"
                )

            mask = dataframe[self.matches_column].apply(self._has_matches)
            filtered = dataframe.loc[mask].reset_index(drop=True)

            context = dict(context)
            context[self.context_key] = filtered
            progress.update(1)
            return context

    @staticmethod
    def _has_matches(value: Any) -> bool:
        return isinstance(value, list) and len(value) > 0
