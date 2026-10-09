"""Registry of asset loaders, keyed by format.

Mirrors the metric registry: loaders register themselves (usually via
``@register_loader``) so the service can resolve a format to a loader. A new
source (Parquet, a Databricks query) is added by registering another loader —
no change to the ingestion pipeline.
"""

from __future__ import annotations

from typing import Callable

from agentspect.assets.base import AssetLoader


class LoaderRegistry:
    def __init__(self) -> None:
        self._loaders: dict[str, type[AssetLoader]] = {}

    def register(
        self,
        cls: type[AssetLoader] | None = None,
        *,
        format: str | None = None,
        replace: bool = False,
    ) -> type[AssetLoader] | Callable[[type[AssetLoader]], type[AssetLoader]]:
        def decorator(target: type[AssetLoader]) -> type[AssetLoader]:
            fmt = format or getattr(target, "format", None)
            if not fmt:
                raise ValueError(
                    f"Loader {target.__name__} must define a `format` (or pass format=...)"
                )
            if fmt in self._loaders and not replace:
                raise ValueError(
                    f"A loader for format '{fmt}' is already registered; pass replace=True"
                )
            if format:
                target.format = format
            self._loaders[fmt] = target
            return target

        return decorator(cls) if cls is not None else decorator

    def get(self, format: str) -> AssetLoader:
        if format not in self._loaders:
            raise KeyError(
                f"No loader for format '{format}'. Known: {sorted(self._loaders)}"
            )
        return self._loaders[format]()

    def formats(self) -> list[str]:
        return list(self._loaders.keys())

    def __contains__(self, format: object) -> bool:
        return format in self._loaders


loader_registry = LoaderRegistry()


def register_loader(
    cls: type[AssetLoader] | None = None,
    *,
    format: str | None = None,
    replace: bool = False,
) -> type[AssetLoader] | Callable[[type[AssetLoader]], type[AssetLoader]]:
    """Register an asset loader into the default registry."""
    return loader_registry.register(cls, format=format, replace=replace)
