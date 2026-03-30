from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

ModuleT = TypeVar("ModuleT")
ModuleType = type[ModuleT]

_MODULE_REGISTRY: dict[str, ModuleType[Any]] = {}


def register_module(
    name: str = "",
) -> Callable[[ModuleType[ModuleT]], ModuleType[ModuleT]]:
    def decorator(module_cls: ModuleType[ModuleT]) -> ModuleType[ModuleT]:
        module_name = name or module_cls.__name__

        if module_name in _MODULE_REGISTRY:
            raise ValueError(f"Module {module_name!r} is already registered")

        _MODULE_REGISTRY[module_name] = module_cls
        return module_cls

    return decorator


def get_module(name: str, **kwargs: Any) -> Any:
    try:
        module_cls = _MODULE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_MODULE_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown module {name!r}. Registered modules: {available}"
        ) from exc

    return module_cls(**kwargs)


from . import models as models
from . import transforms as transforms
from .transforms import get_transform as get_transform

__all__ = ["get_module", "get_transform", "models", "register_module", "transforms"]


try:
    from . import contrastive as _contrastive
except ModuleNotFoundError:
    _contrastive = None

try:
    from . import classification as _classification
except ModuleNotFoundError:
    _classification = None

try:
    from . import graded as _graded
except ModuleNotFoundError:
    _graded = None
