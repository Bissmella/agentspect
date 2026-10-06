"""Built-in metrics, registered into the default registry on import.

The six original aggregate metrics are thin wrappers around the pure functions in
``core`` (so their logic and output shape are unchanged). ``LatencyMetric`` and
``TurnErrorRateMetric`` demonstrate the hook API: they accumulate per-turn state
in ``on_turn`` and emit it in ``compute``.
"""

from __future__ import annotations

from pydantic import BaseModel

from ata.metrics.base import Metric, MetricContext
from ata.metrics.core import (
    compute_boundary_adherence,
    compute_constraint_violations,
    compute_conversation_efficiency,
    compute_recovery_behavior,
    compute_task_completion,
    compute_verification_rate,
)
from ata.metrics.registry import register
from ata.models.suite import Scenario, ScenarioVerdict
from ata.models.transcript import AgentFailure, Transcript, Turn


@register
class TaskCompletion(Metric):
    name = "task_completion"
    description = "Fraction of positive scenarios the agent completed."

    def compute(self, ctx: MetricContext):
        return compute_task_completion(ctx.scenarios, ctx.verdicts)


@register
class BoundaryAdherence(Metric):
    name = "boundary_adherence"
    description = "Fraction of negative scenarios the agent correctly refused."

    def compute(self, ctx: MetricContext):
        return compute_boundary_adherence(ctx.scenarios, ctx.verdicts)


@register
class VerificationRate(Metric):
    name = "verification_rate"
    description = "Fraction of probe-verified successes that held on re-check."

    def compute(self, ctx: MetricContext):
        return compute_verification_rate(ctx.scenarios, ctx.verdicts)


@register
class ConstraintViolations(Metric):
    name = "constraint_violations"
    description = "Per-constraint violation rate, most-violated first."

    def compute(self, ctx: MetricContext):
        return compute_constraint_violations(ctx.scenarios, ctx.verdicts)


@register
class RecoveryBehavior(Metric):
    name = "recovery_behavior"
    description = "Quality breakdown of correct refusals (clean vs. confused/error/leak)."

    def compute(self, ctx: MetricContext):
        return compute_recovery_behavior(ctx.scenarios, ctx.verdicts)


@register
class ConversationEfficiency(Metric):
    name = "conversation_efficiency"
    description = "Average turns to completion / to refusal."

    def compute(self, ctx: MetricContext):
        return compute_conversation_efficiency(ctx.scenarios, ctx.verdicts, ctx.transcripts)



def _percentile(sorted_values: list[int], p: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    k = (len(sorted_values) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


class LatencyResult(BaseModel):
    turns: int
    avg_ms: float | None
    min_ms: int | None
    max_ms: int | None
    p50_ms: float | None
    p95_ms: float | None
    p99_ms: float | None


@register
class LatencyMetric(Metric):
    name = "latency"
    description = "Agent response latency across all non-error turns (ms)."

    def __init__(self) -> None:
        self._samples: list[int] = []

    def on_turn(self, ctx: MetricContext, scenario: Scenario, turn: Turn, index: int) -> None:
        if turn.error is None:
            self._samples.append(turn.latency_ms)

    def compute(self, ctx: MetricContext) -> LatencyResult:
        s = sorted(self._samples)
        n = len(s)
        return LatencyResult(
            turns=n,
            avg_ms=sum(s) / n if n else None,
            min_ms=s[0] if n else None,
            max_ms=s[-1] if n else None,
            p50_ms=_percentile(s, 50),
            p95_ms=_percentile(s, 95),
            p99_ms=_percentile(s, 99),
        )


class TurnErrorRateResult(BaseModel):
    total_turns: int
    error_turns: int
    rate: float


@register
class TurnErrorRateMetric(Metric):
    name = "turn_error_rate"
    description = "Fraction of turns that returned an adapter/transport error."

    def __init__(self) -> None:
        self._total = 0
        self._errors = 0

    def on_turn(self, ctx: MetricContext, scenario: Scenario, turn: Turn, index: int) -> None:
        self._total += 1
        if turn.error is not None:
            self._errors += 1

    def compute(self, ctx: MetricContext) -> TurnErrorRateResult:
        return TurnErrorRateResult(
            total_turns=self._total,
            error_turns=self._errors,
            rate=self._errors / self._total if self._total else 0.0,
        )



class TimeToFirstAudioResult(BaseModel):
    turns: int
    avg_ms: float | None
    p50_ms: float | None
    p95_ms: float | None


@register
class TimeToFirstAudioMetric(Metric):
    """
    Time passed between user speach (ATA) and agent under test voice response.
    """

    name = "time_to_first_audio"
    description = "Agent time-to-first-audio across voice turns (ms)."

    def __init__(self) -> None:
        self._samples: list[int] = []

    def on_turn(self, ctx: MetricContext, scenario: Scenario, turn: Turn, index: int) -> None:
        if turn.error is None and turn.voice and turn.voice.time_to_first_audio_ms is not None:
            self._samples.append(turn.voice.time_to_first_audio_ms)

    def compute(self, ctx: MetricContext) -> TimeToFirstAudioResult:
        s = sorted(self._samples)
        n = len(s)
        return TimeToFirstAudioResult(
            turns=n,
            avg_ms=sum(s) / n if n else None,
            p50_ms=_percentile(s, 50),
            p95_ms=_percentile(s, 95),
        )


#general agent failure metrics

class NoResponseRateResult(BaseModel):
    total_turns: int
    no_response_turns: int
    rate: float


@register
class NoResponseRateMetric(Metric):
    """Fraction of turns where the agent produced no response at all.

    General: a text agent returning empty and a voice agent staying silent both
    surface as ``AgentFailure.NO_RESPONSE``.
    """

    name = "no_response_rate"
    description = "Fraction of turns the agent gave no response."

    def __init__(self) -> None:
        self._total = 0
        self._no_response = 0

    def on_turn(self, ctx: MetricContext, scenario: Scenario, turn: Turn, index: int) -> None:
        self._total += 1
        if turn.agent_failure == AgentFailure.NO_RESPONSE:
            self._no_response += 1

    def compute(self, ctx: MetricContext) -> NoResponseRateResult:
        return NoResponseRateResult(
            total_turns=self._total,
            no_response_turns=self._no_response,
            rate=self._no_response / self._total if self._total else 0.0,
        )


class GreetingRateResult(BaseModel):
    scenarios: int
    greeted: int
    rate: float


@register
class GreetingRateMetric(Metric):
    """Fraction of conversations where the agent spoke first (greeted).

    General: reads ``Transcript.opening_utterance``, which any adapter that
    captures an agent-opening message populates (voice does today).
    """

    name = "greeting_rate"
    description = "Fraction of conversations the agent opened with a greeting."

    def __init__(self) -> None:
        self._scenarios = 0
        self._greeted = 0

    def on_scenario_end(
        self,
        ctx: MetricContext,
        scenario: Scenario,
        transcript: Transcript | None,
        verdict: ScenarioVerdict | None,
    ) -> None:
        if transcript is None:
            return
        self._scenarios += 1
        if transcript.opening_utterance and transcript.opening_utterance.strip():
            self._greeted += 1

    def compute(self, ctx: MetricContext) -> GreetingRateResult:
        return GreetingRateResult(
            scenarios=self._scenarios,
            greeted=self._greeted,
            rate=self._greeted / self._scenarios if self._scenarios else 0.0,
        )


class PrematureDisconnectResult(BaseModel):
    scenarios: int
    disconnected: int
    rate: float


@register
class PrematureDisconnectMetric(Metric):
    """Fraction of conversations the agent dropped before they concluded.

    General: reads ``AgentFailure.DISCONNECTED`` (a voice agent hanging up, or a
    text agent dropping its session mid-conversation).
    """

    name = "premature_disconnect_rate"
    description = "Fraction of conversations the agent ended abruptly."

    def __init__(self) -> None:
        self._scenarios = 0
        self._disconnected = 0

    def on_scenario_end(
        self,
        ctx: MetricContext,
        scenario: Scenario,
        transcript: Transcript | None,
        verdict: ScenarioVerdict | None,
    ) -> None:
        if transcript is None:
            return
        self._scenarios += 1
        if transcript.agent_failure == AgentFailure.DISCONNECTED:
            self._disconnected += 1

    def compute(self, ctx: MetricContext) -> PrematureDisconnectResult:
        return PrematureDisconnectResult(
            scenarios=self._scenarios,
            disconnected=self._disconnected,
            rate=self._disconnected / self._scenarios if self._scenarios else 0.0,
        )


#metrics specific to voice agents

class IntelligibilityResult(BaseModel):
    turns_with_confidence: int
    avg_confidence: float | None
    unintelligible_turns: int


@register
class IntelligibilityMetric(Metric):
    """How intelligible the agent's speech was.

    Averages ``VoiceMeta.stt_confidence`` where the provider reports it, and
    counts turns ATA could not transcribe at all (``AgentFailure.UNINTELLIGIBLE``)
    — a proxy for broken or garbled TTS output.
    """

    name = "intelligibility"
    description = "Agent speech intelligibility (STT confidence + unintelligible turns)."

    def __init__(self) -> None:
        self._confidences: list[float] = []
        self._unintelligible = 0

    def on_turn(self, ctx: MetricContext, scenario: Scenario, turn: Turn, index: int) -> None:
        if turn.agent_failure == AgentFailure.UNINTELLIGIBLE:
            self._unintelligible += 1
        if turn.voice and turn.voice.stt_confidence is not None:
            self._confidences.append(turn.voice.stt_confidence)

    def compute(self, ctx: MetricContext) -> IntelligibilityResult:
        n = len(self._confidences)
        return IntelligibilityResult(
            turns_with_confidence=n,
            avg_confidence=sum(self._confidences) / n if n else None,
            unintelligible_turns=self._unintelligible,
        )


class DeadAirResult(BaseModel):
    turns_with_gaps: int
    total_gaps: int
    max_gap_ms: int | None
    total_dead_air_ms: int


@register
class DeadAirMetric(Metric):
    """Mid-response silence: pauses in the agent's own speech.

    Reads ``VoiceMeta.silence_gaps_ms`` (inter-frame gaps above the adapter's
    dead-air threshold).
    """

    name = "dead_air"
    description = "Mid-response silence gaps in the agent's speech."

    def __init__(self) -> None:
        self._turns_with_gaps = 0
        self._gaps: list[int] = []

    def on_turn(self, ctx: MetricContext, scenario: Scenario, turn: Turn, index: int) -> None:
        if turn.voice and turn.voice.silence_gaps_ms:
            self._turns_with_gaps += 1
            self._gaps.extend(turn.voice.silence_gaps_ms)

    def compute(self, ctx: MetricContext) -> DeadAirResult:
        return DeadAirResult(
            turns_with_gaps=self._turns_with_gaps,
            total_gaps=len(self._gaps),
            max_gap_ms=max(self._gaps) if self._gaps else None,
            total_dead_air_ms=sum(self._gaps),
        )


class AgentSpeechDurationResult(BaseModel):
    turns: int
    avg_ms: float | None
    p50_ms: float | None
    p95_ms: float | None
    total_ms: int


@register
class AgentSpeechDurationMetric(Metric):
    """How long the agent speaks per turn (``VoiceMeta.agent_speech_ms``)."""

    name = "agent_speech_duration"
    description = "Agent speech duration per turn (ms)."

    def __init__(self) -> None:
        self._samples: list[int] = []

    def on_turn(self, ctx: MetricContext, scenario: Scenario, turn: Turn, index: int) -> None:
        if turn.error is None and turn.voice and turn.voice.agent_speech_ms is not None:
            self._samples.append(turn.voice.agent_speech_ms)

    def compute(self, ctx: MetricContext) -> AgentSpeechDurationResult:
        s = sorted(self._samples)
        n = len(s)
        return AgentSpeechDurationResult(
            turns=n,
            avg_ms=sum(s) / n if n else None,
            p50_ms=_percentile(s, 50),
            p95_ms=_percentile(s, 95),
            total_ms=sum(s),
        )



class DataCollectionResult(BaseModel):
    fields_checked: int
    fields_correct: int
    accuracy: float
    avg_cer: float | None


@register
class DataCollectionMetric(Metric):
    """How accurately the agent captured data the user provided (name, email, phone…).

    Aggregates the per-field results produced by ``data_collection`` assertions on
    recall probes: field accuracy and average character/digit error rate. Zero
    fields → empty result (text-only runs are unaffected).
    """

    name = "data_collection_accuracy"
    description = "Field-level accuracy of data the agent collected and recalled."

    def compute(self, ctx: MetricContext) -> DataCollectionResult:
        fields = [f for v in ctx.verdicts.values() for f in v.data_collection]
        n = len(fields)
        correct = sum(1 for f in fields if f.get("match"))
        cers = [f["cer"] for f in fields if f.get("cer") is not None]
        return DataCollectionResult(
            fields_checked=n,
            fields_correct=correct,
            accuracy=correct / n if n else 0.0,
            avg_cer=sum(cers) / len(cers) if cers else None,
        )
