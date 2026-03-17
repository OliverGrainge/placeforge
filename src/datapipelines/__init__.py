from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from .base import Pipeline

PipelineFactory = Callable[[], Pipeline]
PipelineCategory = Literal["train", "val"]

# Maps pipeline_name -> (factory, category)
_PIPELINE_REGISTRY: dict[str, tuple[PipelineFactory, PipelineCategory | None]] = {}


def register_pipeline(
    name: str = "",
    *,
    category: PipelineCategory | None = None,
) -> Callable[[PipelineFactory], PipelineFactory]:
    def decorator(factory: PipelineFactory) -> PipelineFactory:
        pipeline_name = name or factory.__name__

        if pipeline_name in _PIPELINE_REGISTRY:
            raise ValueError(f"Pipeline {pipeline_name!r} is already registered")

        _PIPELINE_REGISTRY[pipeline_name] = (factory, category)
        return factory

    return decorator


def get_pipeline(name: str) -> Pipeline:
    try:
        factory, _ = _PIPELINE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_PIPELINE_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown pipeline {name!r}. Registered pipelines: {available}"
        ) from exc

    return factory()


def list_pipelines(category: PipelineCategory | None = None) -> tuple[str, ...]:
    """Return registered pipeline names, optionally filtered by category."""
    return tuple(
        sorted(
            name
            for name, (_, cat) in _PIPELINE_REGISTRY.items()
            if category is None or cat == category
        )
    )


# Import pipeline modules to trigger registration side-effects.
from . import sf_xl_small  # noqa: E402, F401f
from . import sf_xl
from . import pitts30k
from . import val


__all__ = ["Pipeline", "get_pipeline", "list_pipelines", "register_pipeline"]
