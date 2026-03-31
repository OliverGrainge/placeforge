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
    from .dinov2 import DinoV2BoQ, DinoV2GeM, DinoV2SALAD

    _MODEL_REGISTRY["dinov2_gem"] = DinoV2GeM
    _MODEL_REGISTRY["dinov2_salad"] = DinoV2SALAD
    _MODEL_REGISTRY["dinov2_boq"] = DinoV2BoQ
except ModuleNotFoundError:
    pass
