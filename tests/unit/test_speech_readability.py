"""Speech readability (voice): LLM flags acoustically-clear-but-unintelligible replies."""

from unittest.mock import AsyncMock, MagicMock

from agentspect.agents.scorer import SpeechReadability, _evaluate_speech_readability
from agentspect.metrics.base import MetricContext
from agentspect.metrics.builtin import SpeechReadabilityMetric
from agentspect.models.suite import ScenarioVerdict, Verdict
from agentspect.models.transcript import Transcript, Turn, VoiceMeta


def _voice_transcript(*agent_lines: str) -> Transcript:
    t = Transcript(scenario_id="s1", session_id="x", protocol="voice_websocket")
    for line in agent_lines:
        t.add_turn(Turn(user_message="…", agent_response=line, voice=VoiceMeta()))
    return t


async def test_flags_garbled_timestamp():
    llm = MagicMock()
    llm.chat_with_structured_output = AsyncMock(
        return_value=SpeechReadability(unreadable_turns=1, examples=["2026 05 20 t 10 K00"])
    )
    out = await _evaluate_speech_readability(
        _voice_transcript("Hello, how can I help?", "Your slot is 2026 05 20 t 10 K00"), llm
    )
    assert out == {"agent_turns": 2, "unreadable_turns": 1, "examples": ["2026 05 20 t 10 K00"]}


async def test_unreadable_count_clamped_to_turns():
    llm = MagicMock()
    llm.chat_with_structured_output = AsyncMock(
        return_value=SpeechReadability(unreadable_turns=9, examples=[])  # model over-counts
    )
    out = await _evaluate_speech_readability(_voice_transcript("only one reply"), llm)
    assert out["agent_turns"] == 1 and out["unreadable_turns"] == 1


async def test_none_when_no_agent_speech():
    llm = MagicMock()
    llm.chat_with_structured_output = AsyncMock()
    assert await _evaluate_speech_readability(_voice_transcript("", "  "), llm) is None
    llm.chat_with_structured_output.assert_not_called()  # no LLM call when nothing to judge


def test_metric_aggregates_across_scenarios():
    v1 = ScenarioVerdict(scenario_id="a", verdict=Verdict.SUCCESS, reason="",
                         speech_readability={"agent_turns": 3, "unreadable_turns": 1, "examples": ["x"]})
    v2 = ScenarioVerdict(scenario_id="b", verdict=Verdict.SUCCESS, reason="",
                         speech_readability={"agent_turns": 2, "unreadable_turns": 0, "examples": []})
    r = SpeechReadabilityMetric().compute(MetricContext(verdicts={"a": v1, "b": v2}))
    assert r.turns_checked == 5 and r.unreadable_turns == 1
    assert r.readable_rate == 0.8 and r.examples == ["x"]


def test_metric_na_without_voice():
    v = ScenarioVerdict(scenario_id="s1", verdict=Verdict.SUCCESS, reason="")  # chat run
    r = SpeechReadabilityMetric().compute(MetricContext(verdicts={"s1": v}))
    assert r.turns_checked == 0 and r.readable_rate is None
