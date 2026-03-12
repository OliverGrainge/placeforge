from __future__ import annotations

from collections.abc import Callable

from .base import Pipeline

PipelineFactory = Callable[[], Pipeline]

_PIPELINE_REGISTRY: dict[str, PipelineFactory] = {}


def register_pipeline(name: str = "") -> Callable[[PipelineFactory], PipelineFactory]:
    def decorator(factory: PipelineFactory) -> PipelineFactory:
        pipeline_name = name or factory.__name__

        if pipeline_name in _PIPELINE_REGISTRY:
            raise ValueError(f"Pipeline {pipeline_name!r} is already registered")

        _PIPELINE_REGISTRY[pipeline_name] = factory
        return factory

    return decorator


def get_pipeline(name: str) -> Pipeline:
    try:
        factory = _PIPELINE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PIPELINE_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown pipeline {name!r}. Registered pipelines: {available}"
        ) from exc

    return factory()


# Import pipeline modules to ensure side effects (registration) occur on import
from . import sf_xl_small, sf_xl_small_visual


__all__ = ["Pipeline", "get_pipeline", "register_pipeline"]
