from __future__ import annotations

from typing import Any, Iterable, Iterator

try:
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:
    _tqdm = None


class _NullTqdm:
    def __init__(self, iterable: Iterable[Any] | None = None, **_: Any) -> None:
        self.iterable = iterable

    def __enter__(self) -> "_NullTqdm":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    def __iter__(self) -> Iterator[Any]:
        if self.iterable is None:
            return iter(())
        return iter(self.iterable)

    def update(self, _: int = 1) -> None:
        return None

    def set_postfix_str(self, _: str) -> None:
        return None


def tqdm(*args: Any, **kwargs: Any) -> Any:
    if _tqdm is not None:
        return _tqdm(*args, **kwargs)

    iterable = args[0] if args else kwargs.get("iterable")
    return _NullTqdm(iterable=iterable, **kwargs)
