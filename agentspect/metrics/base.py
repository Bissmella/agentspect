"""Base class and shared context for pluggable metrics.

A metric observes a test run through lifecycle hooks and produces a result in
``compute``. Aggregate metrics (e.g. task completion) ignore the hooks and read
everything from the context at ``compute`` time; streaming metrics (e.g. latency)
accumulate state in ``on_turn`` and emit it in ``compute``.

The engine instantiates a fresh metric object per run, so hook state on ``self``
is safe and isolated between runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from agentspect.models.suite import Scenario, ScenarioVerdict
from agentspect.models.transcript import Transcript, Turn


@dataclass
class MetricContext:
    """Read-only view of a run, passed to every metric hook and to ``compute``."""

    scenarios: list[Scenario] = field(default_factory=list)
    verdicts: dict[str, ScenarioVerdict] = field(default_factory=dict)
    transcripts: dict[str, Transcript] = field(default_factory=dict)
    world_state: dict[str, Any] | None = None
    agent_under_test: Any | None = None

    def scenario_map(self) -> dict[str, Scenario]:
        return {s.id: s for s in self.scenarios}


class Metric(ABC):
    """Base class for all metrics.

    Subclasses must set a unique ``name`` and implement ``compute``. Override any
    of the lifecycle hooks to observe the run as it is replayed; they default to
    no-ops.
    """

    name: ClassVar[str]
    description: ClassVar[str] = ""

    def on_suite_start(self, ctx: MetricContext) -> None:
        """Called once before any scenario is visited."""

    def on_scenario_start(self, ctx: MetricContext, scenario: Scenario) -> None:
        """Called before a scenario's turns are visited."""

    def on_turn(
        self, ctx: MetricContext, scenario: Scenario, turn: Turn, index: int
    ) -> None:
        """Called for each turn in a scenario, in order. Hook site for latency etc."""

    def on_scenario_end(
        self,
        ctx: MetricContext,
        scenario: Scenario,
        transcript: Transcript | None,
        verdict: ScenarioVerdict | None,
    ) -> None:
        """Called after a scenario's turns, with its transcript and verdict."""

    @abstractmethod
    def compute(self, ctx: MetricContext) -> Any:
        """Return the metric result: a pydantic model, dict, list, or primitive."""
