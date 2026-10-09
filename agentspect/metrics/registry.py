"""Registry of available metrics.

Metrics register themselves (usually via the ``@register`` decorator) so the
engine can discover them by name. A default global ``registry`` holds the
built-ins; callers can also build their own ``MetricRegistry`` for isolation.
"""

from __future__ import annotations

from typing import Callable

from agentspect.metrics.base import Metric


class MetricRegistry:
    def __init__(self) -> None:
        self._metrics: dict[str, type[Metric]] = {}

    def register(
        self,
        cls: type[Metric] | None = None,
        *,
        name: str | None = None,
        replace: bool = False,
    ) -> type[Metric] | Callable[[type[Metric]], type[Metric]]:
        """Register a metric class. Usable directly or as a decorator.

        ``@register`` or ``@register(name="...")`` on a ``Metric`` subclass, or
        ``registry.register(MyMetric)``.
        """

        def decorator(target: type[Metric]) -> type[Metric]:
            metric_name = name or getattr(target, "name", None)
            if not metric_name:
                raise ValueError(
                    f"Metric {target.__name__} must define a `name` (or pass name=...)"
                )
            if metric_name in self._metrics and not replace:
                raise ValueError(
                    f"Metric '{metric_name}' is already registered; pass replace=True to override"
                )
            if name:
                target.name = name
            self._metrics[metric_name] = target
            return target

        return decorator(cls) if cls is not None else decorator

    def unregister(self, name: str) -> None:
        self._metrics.pop(name, None)

    def get(self, name: str) -> type[Metric]:
        if name not in self._metrics:
            raise KeyError(f"No metric registered under '{name}'. Known: {self.names()}")
        return self._metrics[name]

    def all(self) -> list[type[Metric]]:
        return list(self._metrics.values())

    def names(self) -> list[str]:
        return list(self._metrics.keys())

    def __contains__(self, name: object) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)


# Default global registry; built-ins register into this on import.
registry = MetricRegistry()


def register(
    cls: type[Metric] | None = None,
    *,
    name: str | None = None,
    replace: bool = False,
) -> type[Metric] | Callable[[type[Metric]], type[Metric]]:
    """Register a metric into the default global registry."""
    return registry.register(cls, name=name, replace=replace)
