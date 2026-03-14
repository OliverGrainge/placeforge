from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

DataModuleT = TypeVar("DataModuleT")
DataModuleType = type[DataModuleT]

_DATAMODULE_REGISTRY: dict[str, DataModuleType[Any]] = {}


def register_datamodule(
    name: str = "",
) -> Callable[[DataModuleType[DataModuleT]], DataModuleType[DataModuleT]]:
    def decorator(dm_cls: DataModuleType[DataModuleT]) -> DataModuleType[DataModuleT]:
        dm_name = name or dm_cls.__name__

        if dm_name in _DATAMODULE_REGISTRY:
            raise ValueError(f"Datamodule {dm_name!r} is already registered")

        _DATAMODULE_REGISTRY[dm_name] = dm_cls
        return dm_cls

    return decorator


def get_datamodule(name: str, **kwargs: Any) -> Any:
    try:
        dm_cls = _DATAMODULE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DATAMODULE_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown datamodule {name!r}. Registered datamodules: {available}"
        ) from exc

    return dm_cls(**kwargs)


from .datamodule import DataModule

__all__ = ["DataModule", "get_datamodule", "register_datamodule"]
