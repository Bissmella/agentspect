import pytest

from agentspect.metrics import (
    Metric,
    MetricContext,
    MetricEngine,
    MetricRegistry,
    compute_metrics,
)
from agentspect.models.suite import Scenario, ScenarioType, ScenarioVerdict, Verdict
from agentspect.models.transcript import Transcript, Turn


def _scenario(sid: str, stype: str = "positive") -> Scenario:
    return Scenario(id=sid, type=ScenarioType(stype), description="d", turns=["hi"])


def _transcript(sid: str, turns: list[Turn]) -> Transcript:
    return Transcript(scenario_id=sid, session_id="s", protocol="callable", turns=turns)


def _turn(latency_ms: int, error: str | None = None) -> Turn:
    return Turn(user_message="u", agent_response="a", latency_ms=latency_ms, error=error)


def _ctx() -> MetricContext:
    scenarios = [_scenario("a"), _scenario("b", "negative")]
    transcripts = {
        "a": _transcript("a", [_turn(10), _turn(20)]),
        "b": _transcript("b", [_turn(30), _turn(999, error="TIMEOUT")]),
    }
    verdicts = {
        "a": ScenarioVerdict(scenario_id="a", verdict=Verdict.SUCCESS_UNVERIFIED, reason=""),
        "b": ScenarioVerdict(scenario_id="b", verdict=Verdict.SUCCESS, reason=""),
    }
    return MetricContext(scenarios=scenarios, verdicts=verdicts, transcripts=transcripts)


class TestRegistry:
    def test_register_and_get(self):
        reg = MetricRegistry()

        @reg.register
        class MyMetric(Metric):
            name = "mine"

            def compute(self, ctx):
                return {"ok": True}

        assert "mine" in reg
        assert reg.get("mine") is MyMetric
        assert len(reg) == 1

    def test_duplicate_raises(self):
        reg = MetricRegistry()

        @reg.register
        class A(Metric):
            name = "dup"

            def compute(self, ctx):
                return 1

        with pytest.raises(ValueError, match="already registered"):

            @reg.register
            class B(Metric):
                name = "dup"

                def compute(self, ctx):
                    return 2

    def test_replace_allowed(self):
        reg = MetricRegistry()

        @reg.register
        class A(Metric):
            name = "x"

            def compute(self, ctx):
                return 1

        @reg.register(replace=True)
        class B(Metric):
            name = "x"

            def compute(self, ctx):
                return 2

        assert reg.get("x") is B

    def test_missing_name_raises(self):
        reg = MetricRegistry()
        with pytest.raises(ValueError, match="must define a `name`"):

            @reg.register
            class NoName(Metric):
                def compute(self, ctx):
                    return 1

    def test_get_unknown_raises(self):
        reg = MetricRegistry()
        with pytest.raises(KeyError):
            reg.get("nope")


class TestEngineBuiltins:
    def test_default_registry_has_builtins(self):
        result = compute_metrics(**_ctx_kwargs())
        # existing six + new hook metrics
        for key in [
            "task_completion",
            "boundary_adherence",
            "verification_rate",
            "constraint_violations",
            "recovery_behavior",
            "conversation_efficiency",
            "latency",
            "turn_error_rate",
        ]:
            assert key in result, f"missing metric {key}"

    def test_select_subset(self):
        result = compute_metrics(**_ctx_kwargs(), metrics=["latency"])
        assert set(result.keys()) == {"latency"}


class TestLatencyMetric:
    def test_percentiles_and_error_exclusion(self):
        result = compute_metrics(**_ctx_kwargs())["latency"]
        # non-error latencies are 10, 20, 30 (the 999 error turn is excluded)
        assert result["turns"] == 3
        assert result["avg_ms"] == 20
        assert result["min_ms"] == 10
        assert result["max_ms"] == 30
        assert result["p50_ms"] == 20

    def test_empty(self):
        ctx = MetricContext(scenarios=[_scenario("a")], verdicts={}, transcripts={})
        result = MetricEngine(metrics=["latency"]).compute(ctx)["latency"]
        assert result["turns"] == 0
        assert result["avg_ms"] is None
        assert result["p95_ms"] is None


class TestTurnErrorRate:
    def test_counts_errors(self):
        result = compute_metrics(**_ctx_kwargs())["turn_error_rate"]
        assert result["total_turns"] == 4
        assert result["error_turns"] == 1
        assert result["rate"] == 0.25


class TestHookLifecycleAndIsolation:
    def test_on_turn_called_per_turn(self):
        reg = MetricRegistry()

        @reg.register
        class TurnCounter(Metric):
            name = "turn_counter"

            def __init__(self):
                self.count = 0

            def on_turn(self, ctx, scenario, turn, index):
                self.count += 1

            def compute(self, ctx):
                return {"count": self.count}

        ctx = _ctx()
        result = MetricEngine(registry=reg).compute(ctx)
        assert result["turn_counter"]["count"] == 4

    def test_fresh_instance_per_run(self):
        reg = MetricRegistry()

        @reg.register
        class Acc(Metric):
            name = "acc"

            def __init__(self):
                self.seen = 0

            def on_turn(self, ctx, scenario, turn, index):
                self.seen += 1

            def compute(self, ctx):
                return {"seen": self.seen}

        engine = MetricEngine(registry=reg)
        r1 = engine.compute(_ctx())
        r2 = engine.compute(_ctx())
        # second run must not accumulate the first run's turns
        assert r1["acc"]["seen"] == 4
        assert r2["acc"]["seen"] == 4


def _ctx_kwargs():
    ctx = _ctx()
    return {
        "scenarios": ctx.scenarios,
        "verdicts": ctx.verdicts,
        "transcripts": ctx.transcripts,
    }
