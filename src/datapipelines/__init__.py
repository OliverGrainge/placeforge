from __future__ import annotations

from collections.abc import Callable

from .base import Pipeline

PipelineFactory = Callable[[], Pipeline]

# Maps pipeline_name -> (factory, category)
_PIPELINE_REGISTRY: dict[str, tuple[PipelineFactory, str | None]] = {}


def register_pipeline(name: str = "") -> Callable[[PipelineFactory], PipelineFactory]:
    def decorator(factory: PipelineFactory) -> PipelineFactory:
        pipeline_name = name or factory.__name__

        if pipeline_name in _PIPELINE_REGISTRY:
            raise ValueError(f"Pipeline {pipeline_name!r} is already registered")

        # Infer category from the subpackage the factory lives in.
        # datapipelines.train.* -> "train", datapipelines.val.* -> "val"
        module_parts = (getattr(factory, "__module__", "") or "").split(".")
        category: str | None = None
        for part in ("train", "val"):
            if part in module_parts:
                category = part
                break

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


def list_pipelines(category: str | None = None) -> tuple[str, ...]:
    """Return registered pipeline names, optionally filtered by category."""
    return tuple(
        sorted(
            name
            for name, (_, cat) in _PIPELINE_REGISTRY.items()
            if category is None or cat == category
        )
    )


# Import pipeline packages to trigger registration side-effects.
# Train pipelines live in datapipelines/train/, val in datapipelines/val/.
from . import train  # noqa: E402, F401
try:
    from . import val  # noqa: E402, F401
except ImportError:
    pass  # val pipelines require extra steps (splitids, geomatches, etc.)


__all__ = ["Pipeline", "get_pipeline", "list_pipelines", "register_pipeline"]
