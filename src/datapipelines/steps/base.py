from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStep(ABC):
    pbar = None

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError