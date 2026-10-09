"""Drives metrics over a run and collects their results.

The engine replays the finished run — scenarios, then each scenario's turns, then
its verdict — through every selected metric's lifecycle hooks, then calls
``compute`` and serialises each result. The same hook API would work if metrics
were later driven live during execution; only the call site changes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentspect.metrics.base import Metric, MetricContext
from agentspect.metrics.registry import MetricRegistry
from agentspect.metrics.registry import registry as default_registry
from agentspect.models.suite import Scenario, ScenarioVerdict
from agentspect.models.transcript import Transcript


def _serialize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


class MetricEngine:
    def __init__(
        self,
        registry: MetricRegistry | None = None,
        metrics: list[str] | None = None,
    ) -> None:
        self.registry = registry or default_registry
        self._selected = metrics

    def _classes(self) -> list[type[Metric]]:
        if self._selected is None:
            return self.registry.all()
        return [self.registry.get(name) for name in self._selected]

    def compute(self, ctx: MetricContext) -> dict[str, Any]:
        instances = [cls() for cls in self._classes()]

        for m in instances:
            m.on_suite_start(ctx)

        for scenario in ctx.scenarios:
            for m in instances:
                m.on_scenario_start(ctx, scenario)

            transcript = ctx.transcripts.get(scenario.id)
            if transcript is not None:
                for index, turn in enumerate(transcript.turns):
                    for m in instances:
                        m.on_turn(ctx, scenario, turn, index)

            verdict = ctx.verdicts.get(scenario.id)
            for m in instances:
                m.on_scenario_end(ctx, scenario, transcript, verdict)

        results: dict[str, Any] = {}
        for m in instances:
            results[m.name] = _serialize(m.compute(ctx))
        return results


def compute_metrics(
    scenarios: list[Scenario],
    verdicts: dict[str, ScenarioVerdict],
    transcripts: dict[str, Transcript],
    *,
    registry: MetricRegistry | None = None,
    metrics: list[str] | None = None,
    **context_kwargs: Any,
) -> dict[str, Any]:
    """Run all (or the named) registered metrics and return ``{name: result}``.

    ``metrics`` selects a subset by name; extra keyword args populate the
    ``MetricContext`` (e.g. ``agent_under_test=...``, ``world_state=...``).
    """
    ctx = MetricContext(
        scenarios=scenarios,
        verdicts=verdicts,
        transcripts=transcripts,
        **context_kwargs,
    )
    return MetricEngine(registry, metrics).compute(ctx)
