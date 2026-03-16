from __future__ import annotations

from typing import Any, Iterable

from tqdm import tqdm

from .steps.base import BaseStep


class Pipeline:
    def __init__(self, name: str, steps: Iterable[BaseStep]) -> None:
        self.name = name
        self.steps = list(steps)

    def run(self, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}

        for step in self.steps:
            name = type(step).__name__
            with tqdm(total=1, desc=name, leave=True) as pbar:
                step.pbar = pbar
                result = step.run(context)
                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
            step.pbar = None
            if result is not None:
                context = result

        return context
