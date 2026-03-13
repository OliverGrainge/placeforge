from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

ModelT = TypeVar("ModelT")
ModelType = type[ModelT]

_MODEL_REGISTRY: dict[str, ModelType[Any]] = {}


def register_model(
    name: str = "",
) -> Callable[[ModelType[ModelT]], ModelType[ModelT]]:
    def decorator(model_cls: ModelType[ModelT]) -> ModelType[ModelT]:
        model_name = name or model_cls.__name__

        if model_name in _MODEL_REGISTRY:
            raise ValueError(f"Model {model_name!r} is already registered")

        _MODEL_REGISTRY[model_name] = model_cls
        return model_cls

    return decorator


def get_model(name: str, **kwargs: Any) -> Any:
    try:
        model_cls = _MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown model {name!r}. Registered models: {available}"
        ) from exc

    return model_cls(**kwargs)


__all__ = ["get_model", "register_model"]


try:
    from . import dinov2 as _dinov2
except ModuleNotFoundError:
    _dinov2 = None

try:
    from . import resnet50 as _resnet50
except ModuleNotFoundError:
    _resnet50 = None
