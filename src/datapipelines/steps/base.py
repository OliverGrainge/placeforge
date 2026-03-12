from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .._progress import tqdm


class BaseStep(ABC):
    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__
        self._pipeline_name = "pipeline"

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError

    def progress(self, total: int | None = None, desc: str | None = None, **kwargs: Any) -> tqdm:
        label = desc or self.name
        return tqdm(
            total=total,
            desc=f"{self._pipeline_name} | {label}",
            dynamic_ncols=True,
            leave=False,
            **kwargs,
        )
