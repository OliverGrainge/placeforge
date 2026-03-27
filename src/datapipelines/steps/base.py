from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseStep(ABC):
    pbar = None
    show_pbar: bool = True

    @abstractmethod
    def run(self, context: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError

    def cache_params(self) -> dict[str, Any] | None:
        """Return a JSON-serializable dict of parameters that determine this step's output.

        Returns None to disable caching for this step (the default).
        """
        return None
