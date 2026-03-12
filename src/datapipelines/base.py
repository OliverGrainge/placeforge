from __future__ import annotations

from typing import Any, Iterable

from ._progress import tqdm
from .steps.base import BaseStep


class Pipeline:
    def __init__(self, name: str, steps: Iterable[BaseStep]) -> None:
        self.name = name
        self.steps = list(steps)

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}

        with tqdm(
            self.steps,
            desc=self.name,
            unit="step",
            dynamic_ncols=True,
        ) as progress:
            for step in progress:
                progress.set_postfix_str(step.name)
                step._pipeline_name = self.name
                result = step.run(context)

                if result is not None:
                    context = result

        return context
