import pytest
from datetime import UTC, datetime

from backend.metrics import (
    compute_all_metrics,
    compute_boundary_adherence,
    compute_constraint_violations,
    compute_conversation_efficiency,
    compute_recovery_behavior,
    compute_task_completion,
    compute_verification_rate,
)
from backend.models.suite import (
    BehavioralAssertion,
    DependsOnType,
    Scenario,
    ScenarioType,
    ScenarioVerdict,
    Verdict,
)
from backend.models.transcript import Transcript, Turn


def _scenario(id, type, **kwargs):
    return Scenario(
        id=id,
        type=ScenarioType(type),
        description=f"test {id}",
        turns=["hello"],
        **kwargs,
    )


def _verdict(id, verdict, recovery_quality=None):
    return ScenarioVerdict(
        scenario_id=id,
        verdict=Verdict(verdict),
        reason="test",
        recovery_quality=recovery_quality,
    )


def _transcript(id, num_turns=2):
    return Transcript(
        scenario_id=id,
        session_id="s",
        protocol="http",
        turns=[
            Turn(
                user_message=f"msg {i}",
                agent_response=f"resp {i}",
                timestamp=datetime.now(UTC),
                latency_ms=50,
            )
            for i in range(num_turns)
        ],
    )


class TestTaskCompletion:
    def test_all_success(self):
        scenarios = [_scenario("p1", "positive"), _scenario("p2", "positive")]
        verdicts = {"p1": _verdict("p1", "success"), "p2": _verdict("p2", "success_unverified")}
        result = compute_task_completion(scenarios, verdicts)
        assert result.rate == 1.0
        assert result.successful == 2
        assert result.total == 2

    def test_mixed(self):
        scenarios = [_scenario("p1", "positive"), _scenario("p2", "positive")]
        verdicts = {"p1": _verdict("p1", "success"), "p2": _verdict("p2", "failure")}
        result = compute_task_completion(scenarios, verdicts)
        assert result.rate == 0.5
        assert result.successful == 1

    def test_excludes_probes(self):
        scenarios = [
            _scenario("p1", "positive"),
            _scenario("probe1", "positive", depends_on="p1", depends_on_type=DependsOnType.PROBE),
        ]
        verdicts = {
            "p1": _verdict("p1", "success"),
            "probe1": _verdict("probe1", "success_unverified"),
        }
        result = compute_task_completion(scenarios, verdicts)
        assert result.total == 1
        assert result.successful == 1

    def test_excludes_negative(self):
        scenarios = [_scenario("p1", "positive"), _scenario("n1", "negative")]
        verdicts = {"p1": _verdict("p1", "failure"), "n1": _verdict("n1", "success")}
        result = compute_task_completion(scenarios, verdicts)
        assert result.total == 1
        assert result.successful == 0

    def test_empty(self):
        result = compute_task_completion([], {})
        assert result.rate == 0.0
        assert result.total == 0

    def test_suspect_not_counted_as_success(self):
        scenarios = [_scenario("p1", "positive")]
        verdicts = {"p1": _verdict("p1", "suspect")}
        result = compute_task_completion(scenarios, verdicts)
        assert result.successful == 0


class TestBoundaryAdherence:
    def test_all_refused(self):
        scenarios = [_scenario("n1", "negative"), _scenario("n2", "negative")]
        verdicts = {"n1": _verdict("n1", "success"), "n2": _verdict("n2", "success")}
        result = compute_boundary_adherence(scenarios, verdicts)
        assert result.rate == 1.0
        assert result.successful == 2

    def test_some_failed(self):
        scenarios = [_scenario("n1", "negative"), _scenario("n2", "negative")]
        verdicts = {"n1": _verdict("n1", "success"), "n2": _verdict("n2", "failure")}
        result = compute_boundary_adherence(scenarios, verdicts)
        assert result.rate == 0.5

    def test_excludes_probes(self):
        scenarios = [
            _scenario("n1", "negative"),
            _scenario("dp1", "negative", depends_on="n1", depends_on_type=DependsOnType.DEFENSIVE_PROBE),
        ]
        verdicts = {"n1": _verdict("n1", "success"), "dp1": _verdict("dp1", "success")}
        result = compute_boundary_adherence(scenarios, verdicts)
        assert result.total == 1

    def test_empty(self):
        result = compute_boundary_adherence([], {})
        assert result.rate == 0.0
        assert result.total == 0


class TestVerificationRate:
    def test_all_confirmed(self):
        scenarios = [
            _scenario("p1", "positive"),
            _scenario("probe1", "positive", depends_on="p1", depends_on_type=DependsOnType.PROBE),
        ]
        verdicts = {
            "p1": _verdict("p1", "success"),
            "probe1": _verdict("probe1", "failure"),
        }
        result = compute_verification_rate(scenarios, verdicts)
        assert result.rate == 1.0
        assert result.confirmed == 1
        assert result.total == 1

    def test_some_suspect(self):
        scenarios = [
            _scenario("p1", "positive"),
            _scenario("probe1", "positive", depends_on="p1", depends_on_type=DependsOnType.PROBE),
            _scenario("p2", "positive"),
            _scenario("probe2", "positive", depends_on="p2", depends_on_type=DependsOnType.PROBE),
        ]
        verdicts = {
            "p1": _verdict("p1", "success"),
            "probe1": _verdict("probe1", "failure"),
            "p2": _verdict("p2", "suspect"),
            "probe2": _verdict("probe2", "success_unverified"),
        }
        result = compute_verification_rate(scenarios, verdicts)
        assert result.rate == 0.5
        assert result.confirmed == 1
        assert result.total == 2

    def test_no_probes(self):
        scenarios = [_scenario("p1", "positive")]
        verdicts = {"p1": _verdict("p1", "success_unverified")}
        result = compute_verification_rate(scenarios, verdicts)
        assert result.rate == 0.0
        assert result.total == 0


class TestConstraintViolations:
    def test_per_constraint_breakdown(self):
        scenarios = [
            _scenario("n1", "negative", target_constraint="only verified users can book"),
            _scenario("n2", "negative", target_constraint="only verified users can book"),
            _scenario("n3", "negative", target_constraint="slots must be within 09:00-18:00"),
        ]
        verdicts = {
            "n1": _verdict("n1", "failure"),
            "n2": _verdict("n2", "success"),
            "n3": _verdict("n3", "failure"),
        }
        result = compute_constraint_violations(scenarios, verdicts)
        assert len(result) == 2

        by_constraint = {c.constraint: c for c in result}
        v = by_constraint["only verified users can book"]
        assert v.violated == 1
        assert v.tested == 2
        assert v.rate == 0.5

        s = by_constraint["slots must be within 09:00-18:00"]
        assert s.violated == 1
        assert s.tested == 1
        assert s.rate == 1.0

    def test_sorted_by_rate_desc(self):
        scenarios = [
            _scenario("n1", "negative", target_constraint="rule A"),
            _scenario("n2", "negative", target_constraint="rule B"),
            _scenario("n3", "negative", target_constraint="rule B"),
        ]
        verdicts = {
            "n1": _verdict("n1", "failure"),
            "n2": _verdict("n2", "failure"),
            "n3": _verdict("n3", "failure"),
        }
        result = compute_constraint_violations(scenarios, verdicts)
        assert result[0].rate >= result[1].rate

    def test_adversarial_labels(self):
        scenarios = [
            _scenario("n1", "negative", target_constraint="out-of-scope request"),
            _scenario("n2", "negative", target_constraint="prompt injection attempt"),
        ]
        verdicts = {
            "n1": _verdict("n1", "success"),
            "n2": _verdict("n2", "failure"),
        }
        result = compute_constraint_violations(scenarios, verdicts)
        assert len(result) == 2
        by_constraint = {c.constraint: c for c in result}
        assert by_constraint["out-of-scope request"].violated == 0
        assert by_constraint["prompt injection attempt"].violated == 1

    def test_failure_corrupt_counted(self):
        scenarios = [_scenario("n1", "negative", target_constraint="rule A")]
        verdicts = {"n1": _verdict("n1", "failure_corrupt")}
        result = compute_constraint_violations(scenarios, verdicts)
        assert result[0].violated == 1

    def test_excludes_probes(self):
        scenarios = [
            _scenario("n1", "negative", target_constraint="rule A"),
            _scenario("dp1", "negative", depends_on="n1", depends_on_type=DependsOnType.DEFENSIVE_PROBE, target_constraint="rule A"),
        ]
        verdicts = {"n1": _verdict("n1", "success"), "dp1": _verdict("dp1", "success")}
        result = compute_constraint_violations(scenarios, verdicts)
        assert result[0].tested == 1

    def test_empty(self):
        result = compute_constraint_violations([], {})
        assert result == []


class TestRecoveryBehavior:
    def test_all_clean(self):
        scenarios = [_scenario("n1", "negative"), _scenario("n2", "negative")]
        verdicts = {
            "n1": _verdict("n1", "success", recovery_quality="clean_refusal"),
            "n2": _verdict("n2", "success", recovery_quality="clean_refusal"),
        }
        result = compute_recovery_behavior(scenarios, verdicts)
        assert result.rate == 1.0
        assert result.clean_refusal == 2
        assert result.total_adversarial == 2

    def test_mixed(self):
        scenarios = [
            _scenario("n1", "negative"),
            _scenario("n2", "negative"),
            _scenario("n3", "negative"),
        ]
        verdicts = {
            "n1": _verdict("n1", "success", recovery_quality="clean_refusal"),
            "n2": _verdict("n2", "success", recovery_quality="confused_response"),
            "n3": _verdict("n3", "success", recovery_quality="information_leak"),
        }
        result = compute_recovery_behavior(scenarios, verdicts)
        assert result.rate == pytest.approx(1.0 / 3.0)
        assert result.clean_refusal == 1
        assert result.confused_response == 1
        assert result.information_leak == 1

    def test_unclassified(self):
        scenarios = [_scenario("n1", "negative")]
        verdicts = {"n1": _verdict("n1", "success", recovery_quality=None)}
        result = compute_recovery_behavior(scenarios, verdicts)
        assert result.unclassified == 1
        assert result.rate == 0.0

    def test_excludes_failed_negatives(self):
        scenarios = [_scenario("n1", "negative"), _scenario("n2", "negative")]
        verdicts = {
            "n1": _verdict("n1", "success", recovery_quality="clean_refusal"),
            "n2": _verdict("n2", "failure"),
        }
        result = compute_recovery_behavior(scenarios, verdicts)
        assert result.total_adversarial == 1

    def test_empty(self):
        result = compute_recovery_behavior([], {})
        assert result.rate == 0.0
        assert result.total_adversarial == 0


class TestConversationEfficiency:
    def test_positive_avg(self):
        scenarios = [_scenario("p1", "positive"), _scenario("p2", "positive")]
        verdicts = {
            "p1": _verdict("p1", "success"),
            "p2": _verdict("p2", "success_unverified"),
        }
        transcripts = {"p1": _transcript("p1", 3), "p2": _transcript("p2", 5)}
        result = compute_conversation_efficiency(scenarios, verdicts, transcripts)
        assert result.avg_turns_to_completion == 4.0

    def test_negative_avg(self):
        scenarios = [_scenario("n1", "negative"), _scenario("n2", "negative")]
        verdicts = {
            "n1": _verdict("n1", "success"),
            "n2": _verdict("n2", "success"),
        }
        transcripts = {"n1": _transcript("n1", 2), "n2": _transcript("n2", 1)}
        result = compute_conversation_efficiency(scenarios, verdicts, transcripts)
        assert result.avg_turns_to_refusal == 1.5

    def test_excludes_failed(self):
        scenarios = [_scenario("p1", "positive"), _scenario("p2", "positive")]
        verdicts = {
            "p1": _verdict("p1", "success"),
            "p2": _verdict("p2", "failure"),
        }
        transcripts = {"p1": _transcript("p1", 3), "p2": _transcript("p2", 10)}
        result = compute_conversation_efficiency(scenarios, verdicts, transcripts)
        assert result.avg_turns_to_completion == 3.0

    def test_no_data_returns_none(self):
        result = compute_conversation_efficiency([], {}, {})
        assert result.avg_turns_to_completion is None
        assert result.avg_turns_to_refusal is None

    def test_excludes_probes(self):
        scenarios = [
            _scenario("p1", "positive"),
            _scenario("probe1", "positive", depends_on="p1", depends_on_type=DependsOnType.PROBE),
        ]
        verdicts = {
            "p1": _verdict("p1", "success"),
            "probe1": _verdict("probe1", "success_unverified"),
        }
        transcripts = {
            "p1": _transcript("p1", 3),
            "probe1": _transcript("probe1", 1),
        }
        result = compute_conversation_efficiency(scenarios, verdicts, transcripts)
        assert result.avg_turns_to_completion == 3.0


class TestComputeAllMetrics:
    def test_integration(self):
        scenarios = [
            _scenario("p1", "positive"),
            _scenario("p2", "positive"),
            _scenario("probe1", "positive", depends_on="p1", depends_on_type=DependsOnType.PROBE),
            _scenario("n1", "negative", target_constraint="only verified users can book"),
            _scenario("n2", "negative", target_constraint="out-of-scope request"),
        ]
        verdicts = {
            "p1": _verdict("p1", "success"),
            "p2": _verdict("p2", "failure"),
            "probe1": _verdict("probe1", "failure"),
            "n1": _verdict("n1", "success", recovery_quality="clean_refusal"),
            "n2": _verdict("n2", "failure"),
        }
        transcripts = {
            "p1": _transcript("p1", 3),
            "p2": _transcript("p2", 5),
            "probe1": _transcript("probe1", 1),
            "n1": _transcript("n1", 2),
            "n2": _transcript("n2", 4),
        }

        metrics = compute_all_metrics(scenarios, verdicts, transcripts)

        assert metrics.task_completion.rate == 0.5
        assert metrics.task_completion.total == 2
        assert metrics.boundary_adherence.rate == 0.5
        assert metrics.boundary_adherence.total == 2
        assert metrics.verification_rate.confirmed == 1
        assert metrics.verification_rate.total == 1
        assert len(metrics.constraint_violations) == 2
        assert metrics.recovery_behavior.clean_refusal == 1
        assert metrics.recovery_behavior.total_adversarial == 1
        assert metrics.conversation_efficiency.avg_turns_to_completion == 3.0
        assert metrics.conversation_efficiency.avg_turns_to_refusal == 2.0

    def test_serializes_to_dict(self):
        scenarios = [_scenario("p1", "positive")]
        verdicts = {"p1": _verdict("p1", "success")}
        transcripts = {"p1": _transcript("p1", 2)}
        metrics = compute_all_metrics(scenarios, verdicts, transcripts)
        d = metrics.model_dump()
        assert "task_completion" in d
        assert "boundary_adherence" in d
        assert "verification_rate" in d
        assert "constraint_violations" in d
        assert "recovery_behavior" in d
        assert "conversation_efficiency" in d
