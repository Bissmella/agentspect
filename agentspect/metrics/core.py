from collections import Counter

from pydantic import BaseModel

from agentspect.models.suite import (
    DependsOnType,
    Scenario,
    ScenarioType,
    ScenarioVerdict,
    Verdict,
)
from agentspect.models.transcript import Transcript


class TaskCompletionMetric(BaseModel):
    rate: float
    successful: int
    total: int


class BoundaryAdherenceMetric(BaseModel):
    rate: float
    successful: int
    total: int


class VerificationRateMetric(BaseModel):
    rate: float
    confirmed: int
    total: int


class ConstraintViolation(BaseModel):
    constraint: str
    violated: int
    tested: int
    rate: float


class RecoveryBehaviorMetric(BaseModel):
    rate: float
    total_adversarial: int
    clean_refusal: int
    confused_response: int
    error_response: int
    information_leak: int
    unclassified: int


class ConversationEfficiencyMetric(BaseModel):
    avg_turns_to_completion: float | None
    avg_turns_to_refusal: float | None


class QuantitativeMetrics(BaseModel):
    task_completion: TaskCompletionMetric
    boundary_adherence: BoundaryAdherenceMetric
    verification_rate: VerificationRateMetric
    constraint_violations: list[ConstraintViolation]
    recovery_behavior: RecoveryBehaviorMetric
    conversation_efficiency: ConversationEfficiencyMetric


def _is_probe(scenario: Scenario) -> bool:
    return scenario.depends_on is not None and scenario.depends_on_type is not None


def compute_task_completion(
    scenarios: list[Scenario],
    verdicts: dict[str, ScenarioVerdict],
) -> TaskCompletionMetric:
    positive = [s for s in scenarios if s.type == ScenarioType.POSITIVE and not _is_probe(s)]
    total = len(positive)
    successful = sum(
        1 for s in positive
        if s.id in verdicts and verdicts[s.id].verdict in (Verdict.SUCCESS, Verdict.SUCCESS_UNVERIFIED)
    )
    return TaskCompletionMetric(
        rate=successful / total if total > 0 else 0.0,
        successful=successful,
        total=total,
    )


def compute_boundary_adherence(
    scenarios: list[Scenario],
    verdicts: dict[str, ScenarioVerdict],
) -> BoundaryAdherenceMetric:
    negative = [s for s in scenarios if s.type == ScenarioType.NEGATIVE and not _is_probe(s)]
    total = len(negative)
    successful = sum(
        1 for s in negative
        if s.id in verdicts and verdicts[s.id].verdict == Verdict.SUCCESS
    )
    return BoundaryAdherenceMetric(
        rate=successful / total if total > 0 else 0.0,
        successful=successful,
        total=total,
    )


def compute_verification_rate(
    scenarios: list[Scenario],
    verdicts: dict[str, ScenarioVerdict],
) -> VerificationRateMetric:
    probes = [s for s in scenarios if s.depends_on_type == DependsOnType.PROBE]
    total = len(probes)
    confirmed = 0
    for probe in probes:
        parent_id = probe.depends_on
        if parent_id and parent_id in verdicts:
            if verdicts[parent_id].verdict == Verdict.SUCCESS:
                confirmed += 1
    return VerificationRateMetric(
        rate=confirmed / total if total > 0 else 0.0,
        confirmed=confirmed,
        total=total,
    )


def compute_constraint_violations(
    scenarios: list[Scenario],
    verdicts: dict[str, ScenarioVerdict],
) -> list[ConstraintViolation]:
    negative = [
        s for s in scenarios
        if s.type == ScenarioType.NEGATIVE and not _is_probe(s) and s.target_constraint
    ]

    by_constraint: dict[str, list[Scenario]] = {}
    for s in negative:
        by_constraint.setdefault(s.target_constraint, []).append(s)

    results = []
    for constraint, group in by_constraint.items():
        tested = len(group)
        violated = sum(
            1 for s in group
            if s.id in verdicts and verdicts[s.id].verdict in (Verdict.FAILURE, Verdict.FAILURE_CORRUPT)
        )
        results.append(ConstraintViolation(
            constraint=constraint,
            violated=violated,
            tested=tested,
            rate=violated / tested if tested > 0 else 0.0,
        ))

    results.sort(key=lambda c: c.rate, reverse=True)
    return results


def compute_recovery_behavior(
    scenarios: list[Scenario],
    verdicts: dict[str, ScenarioVerdict],
) -> RecoveryBehaviorMetric:
    negative_success = [
        s for s in scenarios
        if s.type == ScenarioType.NEGATIVE
        and not _is_probe(s)
        and s.id in verdicts
        and verdicts[s.id].verdict == Verdict.SUCCESS
    ]

    total = len(negative_success)
    counts: Counter[str] = Counter()
    for s in negative_success:
        quality = verdicts[s.id].recovery_quality
        if quality:
            counts[quality] += 1
        else:
            counts["unclassified"] += 1

    clean = counts.get("clean_refusal", 0)
    return RecoveryBehaviorMetric(
        rate=clean / total if total > 0 else 0.0,
        total_adversarial=total,
        clean_refusal=clean,
        confused_response=counts.get("confused_response", 0),
        error_response=counts.get("error_response", 0),
        information_leak=counts.get("information_leak", 0),
        unclassified=counts.get("unclassified", 0),
    )


def compute_conversation_efficiency(
    scenarios: list[Scenario],
    verdicts: dict[str, ScenarioVerdict],
    transcripts: dict[str, Transcript],
) -> ConversationEfficiencyMetric:
    positive_turns = []
    negative_turns = []

    for s in scenarios:
        if _is_probe(s) or s.id not in verdicts or s.id not in transcripts:
            continue

        v = verdicts[s.id]
        turn_count = len(transcripts[s.id].turns)

        if s.type == ScenarioType.POSITIVE and v.verdict in (Verdict.SUCCESS, Verdict.SUCCESS_UNVERIFIED):
            positive_turns.append(turn_count)
        elif s.type == ScenarioType.NEGATIVE and v.verdict == Verdict.SUCCESS:
            negative_turns.append(turn_count)

    return ConversationEfficiencyMetric(
        avg_turns_to_completion=sum(positive_turns) / len(positive_turns) if positive_turns else None,
        avg_turns_to_refusal=sum(negative_turns) / len(negative_turns) if negative_turns else None,
    )


def compute_all_metrics(
    scenarios: list[Scenario],
    verdicts: dict[str, ScenarioVerdict],
    transcripts: dict[str, Transcript],
) -> QuantitativeMetrics:
    return QuantitativeMetrics(
        task_completion=compute_task_completion(scenarios, verdicts),
        boundary_adherence=compute_boundary_adherence(scenarios, verdicts),
        verification_rate=compute_verification_rate(scenarios, verdicts),
        constraint_violations=compute_constraint_violations(scenarios, verdicts),
        recovery_behavior=compute_recovery_behavior(scenarios, verdicts),
        conversation_efficiency=compute_conversation_efficiency(scenarios, verdicts, transcripts),
    )
