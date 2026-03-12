from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

DatamoduleT = TypeVar("DatamoduleT")
DatamoduleType = type[DatamoduleT]

_DATAMODULE_REGISTRY: dict[str, DatamoduleType[Any]] = {}


def register_datamodule(
    name: str = "",
) -> Callable[[DatamoduleType[DatamoduleT]], DatamoduleType[DatamoduleT]]:
    def decorator(
        datamodule_cls: DatamoduleType[DatamoduleT],
    ) -> DatamoduleType[DatamoduleT]:
        datamodule_name = name or datamodule_cls.__name__

        if datamodule_name in _DATAMODULE_REGISTRY:
            raise ValueError(
                f"Datamodule {datamodule_name!r} is already registered"
            )

        _DATAMODULE_REGISTRY[datamodule_name] = datamodule_cls
        return datamodule_cls

    return decorator


def get_datamodule(name: str, **kwargs: Any) -> Any:
    try:
        datamodule_cls = _DATAMODULE_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_DATAMODULE_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown datamodule {name!r}. Registered datamodules: {available}"
        ) from exc

    return datamodule_cls(**kwargs)


__all__ = ["get_datamodule", "register_datamodule"]


try:
    from . import datamodule as _datamodule
except ModuleNotFoundError:
    _datamodule = None
