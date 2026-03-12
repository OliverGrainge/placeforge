from __future__ import annotations

from typing import Any, Iterable

from .steps.base import BaseStep


class Pipeline:
    def __init__(self, name: str, steps: Iterable[BaseStep]) -> None:
        self.name = name
        self.steps = list(steps)

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}

        for index, step in enumerate(self.steps):
            step._pipeline_name = self.name
            step._progress_position = index
            result = step.run(context)

            if result is not None:
                context = result

        return context
