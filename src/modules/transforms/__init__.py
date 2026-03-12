from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

TransformT = TypeVar("TransformT")
TransformType = type[TransformT]

_TRANSFORM_REGISTRY: dict[str, TransformType[Any]] = {}


def register_transform(
    name: str = "",
) -> Callable[[TransformType[TransformT]], TransformType[TransformT]]:
    def decorator(transform_cls: TransformType[TransformT]) -> TransformType[TransformT]:
        transform_name = name or transform_cls.__name__

        if transform_name in _TRANSFORM_REGISTRY:
            raise ValueError(f"Transform {transform_name!r} is already registered")

        _TRANSFORM_REGISTRY[transform_name] = transform_cls
        return transform_cls

    return decorator


def get_transform(name: str, **kwargs: Any) -> Any:
    try:
        transform_cls = _TRANSFORM_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TRANSFORM_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown transform {name!r}. Registered transforms: {available}"
        ) from exc

    return transform_cls(**kwargs)


__all__ = ["get_transform", "register_transform"]


try:
    from . import image as _image
except ModuleNotFoundError:
    _image = None
