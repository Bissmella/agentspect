"""Tests for the thin voice channel: models, registry/BYO, VoiceIO, adapter, metric."""

import base64
import json

import pytest
from ata.adapters.voice_ws_adapter import VoiceWebSocketAdapter
from ata.adapters.ws_adapter import create_adapter
from ata.metrics.builtin import (
    GreetingRateMetric,
    IntelligibilityMetric,
    NoResponseRateMetric,
    PrematureDisconnectMetric,
    TimeToFirstAudioMetric,
)
from ata.models.suite import Scenario, ScenarioType
from ata.models.transcript import AgentFailure, Transcript, Turn, VoiceMeta
from ata.models.yaml_input import STTSpec, TTSSpec, VoiceConfig
from ata.voice.client import (
    STTClient,
    TranscriptionResult,
    TTSClient,
    VoiceIO,
    VoiceIODefaults,
)
from ata.voice.registry import create_voice_io, register_stt, register_tts

# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeSTT(STTClient):
    def __init__(self, model=None, api_key=None):
        self.model = model

    async def transcribe(self, audio: bytes, *, language=None) -> TranscriptionResult:
        return TranscriptionResult(
            text=audio.decode("utf-8", errors="replace"),
            confidence=0.9,
            language=language,
        )


class FakeTTS(TTSClient):
    def __init__(self, model=None, api_key=None):
        self.model = model

    async def synthesize(self, text, *, voice=None, language=None, accent=None) -> bytes:
        return text.encode("utf-8")


def _fake_voice_io():
    return VoiceIO(
        stt=FakeSTT(),
        tts=FakeTTS(),
        defaults=VoiceIODefaults(voice="alloy", language="en", accent=None),
    )


class FakeConnection:
    """Minimal stand-in for a websockets connection."""

    def __init__(self, outbound: list[str], hang: bool = False, raise_on_recv: bool = False):
        self._outbound = list(outbound)
        self._hang = hang
        self._raise_on_recv = raise_on_recv
        self.sent: list[str] = []
        self.closed = False

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str:
        if self._outbound:
            return self._outbound.pop(0)
        if self._raise_on_recv:
            from websockets.exceptions import ConnectionClosedError

            raise ConnectionClosedError(None, None)
        if self._hang:
            import asyncio

            await asyncio.sleep(10)  # forces the adapter's wait_for to time out
        raise AssertionError("recv called with nothing queued and hang=False")

    async def close(self) -> None:
        self.closed = True


def _audio_frame(text: str) -> str:
    return json.dumps({"type": "audio", "data": base64.b64encode(text.encode()).decode()})


_EOS = json.dumps({"type": "end_of_speech"})


def _adapter_with(conn: FakeConnection, timeout: float = 5.0) -> VoiceWebSocketAdapter:
    adapter = VoiceWebSocketAdapter(url="wss://x", voice_io=_fake_voice_io(), timeout=timeout)

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    return adapter


# ── Models: additive, backward compatible ─────────────────────────────────────

def test_turn_voice_is_optional():
    t = Turn(user_message="hi", agent_response="hello")
    assert t.voice is None


def test_voice_meta_roundtrips():
    vm = VoiceMeta(time_to_first_audio_ms=120, stt_confidence=0.8, voice="alloy", language="en")
    t = Turn(user_message="hi", agent_response="hello", voice=vm)
    assert t.voice.time_to_first_audio_ms == 120
    assert Turn.model_validate(t.model_dump()).voice.voice == "alloy"


# ── Registry / BYO ─────────────────────────────────────────────────────────────

def test_create_voice_io_resolves_registered_providers():
    register_stt("faketest", replace=True)(FakeSTT)
    register_tts("faketest", replace=True)(FakeTTS)
    vio = create_voice_io(stt_provider="faketest", tts_provider="faketest", voice="v", language="fr")
    assert isinstance(vio.stt, FakeSTT)
    assert vio.defaults.voice == "v"
    assert vio.defaults.language == "fr"


def test_create_voice_io_unknown_provider_raises():
    with pytest.raises(KeyError):
        create_voice_io(stt_provider="nope", tts_provider="faketest")


def test_openai_reference_is_registered():
    from ata.voice.registry import stt_registry, tts_registry

    assert "openai" in stt_registry
    assert "openai" in tts_registry


# ── VoiceIO applies defaults ────────────────────────────────────────────────────

async def test_voice_io_synthesize_and_transcribe_roundtrip():
    vio = _fake_voice_io()
    audio = await vio.synthesize("book me a slot")
    result = await vio.transcribe(audio)
    assert result.text == "book me a slot"


# ── Adapter ─────────────────────────────────────────────────────────────────────

async def test_adapter_captures_opening_greeting():
    conn = FakeConnection([_audio_frame("Thanks for calling ACME."), _EOS])
    adapter = _adapter_with(conn)
    session_id = await adapter.start_session("s1")
    transcript = adapter.create_transcript("s1", session_id, "voicewebsocket")
    assert transcript.opening_utterance == "Thanks for calling ACME."
    assert transcript.opening_voice is not None
    await adapter.close()


async def test_adapter_send_turn_roundtrips_and_records_voice_meta():
    # greeting burst, then the reply to the first turn
    conn = FakeConnection([_EOS, _audio_frame("You are booked."), _EOS])
    adapter = _adapter_with(conn)
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "I'd like slot A")
    assert turn.error is None
    assert turn.agent_response == "You are booked."
    assert turn.voice is not None
    assert turn.voice.voice == "alloy"
    assert turn.voice.stt_confidence == 0.9
    assert turn.voice.time_to_first_audio_ms is not None
    # the user's text was synthesized and sent as an audio frame
    assert json.loads(conn.sent[0])["type"] == "audio"
    await adapter.close()


async def test_adapter_silence_is_no_response_not_error():
    # greeting ends, then the agent goes silent → behavioral NO_RESPONSE, not ERROR.
    conn = FakeConnection([_EOS], hang=True)
    adapter = _adapter_with(conn, timeout=0.2)
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello?")
    assert turn.error is None
    assert turn.agent_failure == AgentFailure.NO_RESPONSE
    assert turn.agent_response == ""
    await adapter.close()


async def test_adapter_midcall_disconnect_is_agent_failure():
    # greeting ends, then the connection drops mid-turn → DISCONNECTED (agent hung up).
    conn = FakeConnection([_EOS], raise_on_recv=True)
    adapter = _adapter_with(conn)
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "are you there?")
    assert turn.error is None
    assert turn.agent_failure == AgentFailure.DISCONNECTED
    await adapter.close()


async def test_adapter_unintelligible_when_transcription_empty():
    # greeting ends, then the agent emits audio that transcribes to whitespace.
    blank_audio = _audio_frame("   ")
    conn = FakeConnection([_EOS, blank_audio, _EOS])
    adapter = _adapter_with(conn)
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    assert turn.error is None
    assert turn.agent_failure == AgentFailure.UNINTELLIGIBLE
    assert turn.voice is not None
    await adapter.close()


async def test_adapter_missing_greeting_is_not_an_error():
    conn = FakeConnection([], hang=True)  # never greets
    adapter = VoiceWebSocketAdapter(
        url="wss://x", voice_io=_fake_voice_io(), timeout=5.0, greeting_timeout=0.1
    )

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    transcript = adapter.create_transcript("s1", session_id, "voicewebsocket")
    assert transcript.opening_utterance is None
    await adapter.close()


# ── Wiring ───────────────────────────────────────────────────────────────────────

def test_create_adapter_routes_voice_websocket():
    register_stt("faketest", replace=True)(FakeSTT)
    register_tts("faketest", replace=True)(FakeTTS)
    voice = VoiceConfig(stt=STTSpec(provider="faketest"), tts=TTSSpec(provider="faketest"))
    adapter = create_adapter("voice_websocket", url="wss://x", voice=voice)
    assert isinstance(adapter, VoiceWebSocketAdapter)


# ── Proof metric ──────────────────────────────────────────────────────────────────

def _mk_scenario(sid: str) -> Scenario:
    return Scenario(id=sid, type=ScenarioType.POSITIVE, description="d", turns=["hi"])


def test_time_to_first_audio_metric():
    from ata.metrics.base import MetricContext

    transcript = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket")
    transcript.add_turn(Turn(user_message="a", agent_response="b", voice=VoiceMeta(time_to_first_audio_ms=100)))
    transcript.add_turn(Turn(user_message="c", agent_response="d", voice=VoiceMeta(time_to_first_audio_ms=300)))

    scenario = _mk_scenario("s1")
    ctx = MetricContext(scenarios=[scenario], transcripts={"s1": transcript})

    m = TimeToFirstAudioMetric()
    for i, turn in enumerate(transcript.turns):
        m.on_turn(ctx, scenario, turn, i)
    result = m.compute(ctx)
    assert result.turns == 2
    assert result.avg_ms == 200.0


def test_time_to_first_audio_metric_ignores_text_turns():
    from ata.metrics.base import MetricContext

    transcript = Transcript(scenario_id="s1", session_id="x", protocol="http")
    transcript.add_turn(Turn(user_message="a", agent_response="b"))  # no voice meta
    scenario = _mk_scenario("s1")
    ctx = MetricContext(scenarios=[scenario], transcripts={"s1": transcript})

    m = TimeToFirstAudioMetric()
    m.on_turn(ctx, scenario, transcript.turns[0], 0)
    result = m.compute(ctx)
    assert result.turns == 0
    assert result.avg_ms is None


# ── Backbone: model + failure metrics + scorer routing ─────────────────────────

def test_transcript_agent_failure_property_returns_last():
    t = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket")
    t.add_turn(Turn(user_message="a", agent_response="b"))
    t.add_turn(Turn(user_message="c", agent_response="", agent_failure=AgentFailure.NO_RESPONSE))
    assert t.agent_failure == AgentFailure.NO_RESPONSE

    clean = Transcript(scenario_id="s2", session_id="y", protocol="http")
    clean.add_turn(Turn(user_message="a", agent_response="b"))
    assert clean.agent_failure is None


def _ctx_with(transcript):
    from ata.metrics.base import MetricContext

    scenario = _mk_scenario(transcript.scenario_id)
    return MetricContext(scenarios=[scenario], transcripts={transcript.scenario_id: transcript}), scenario


def _run_metric(metric, transcript):
    ctx, scenario = _ctx_with(transcript)
    metric.on_scenario_start(ctx, scenario)
    for i, turn in enumerate(transcript.turns):
        metric.on_turn(ctx, scenario, turn, i)
    metric.on_scenario_end(ctx, scenario, transcript, None)
    return metric.compute(ctx)


def test_no_response_rate_metric():
    t = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket")
    t.add_turn(Turn(user_message="a", agent_response="ok"))
    t.add_turn(Turn(user_message="b", agent_response="", agent_failure=AgentFailure.NO_RESPONSE))
    result = _run_metric(NoResponseRateMetric(), t)
    assert result.total_turns == 2
    assert result.no_response_turns == 1
    assert result.rate == 0.5


def test_greeting_rate_metric():
    greeted = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket",
                         opening_utterance="Hello, how can I help?")
    greeted.add_turn(Turn(user_message="a", agent_response="b"))
    assert _run_metric(GreetingRateMetric(), greeted).rate == 1.0

    silent = Transcript(scenario_id="s2", session_id="y", protocol="http")
    silent.add_turn(Turn(user_message="a", agent_response="b"))
    assert _run_metric(GreetingRateMetric(), silent).rate == 0.0


def test_premature_disconnect_metric():
    t = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket")
    t.add_turn(Turn(user_message="a", agent_response="b"))
    t.add_turn(Turn(user_message="c", agent_response="", agent_failure=AgentFailure.DISCONNECTED))
    result = _run_metric(PrematureDisconnectMetric(), t)
    assert result.scenarios == 1
    assert result.disconnected == 1
    assert result.rate == 1.0


def test_intelligibility_metric():
    t = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket")
    t.add_turn(Turn(user_message="a", agent_response="b", voice=VoiceMeta(stt_confidence=0.8)))
    t.add_turn(Turn(user_message="c", agent_response="", agent_failure=AgentFailure.UNINTELLIGIBLE))
    result = _run_metric(IntelligibilityMetric(), t)
    assert result.turns_with_confidence == 1
    assert result.avg_confidence == 0.8
    assert result.unintelligible_turns == 1


async def test_scorer_agent_failure_forces_failure_despite_passing_assertions():
    """A behavioral failure must NOT be scored ERROR, and must fail the run even
    when the (mocked) assertion evaluator would say the check passed."""
    from unittest.mock import AsyncMock, MagicMock

    from ata.agents.scorer import AssertionResult, scorer_node
    from ata.models.suite import ScenarioType, TranscriptAssertion, Verdict

    llm = MagicMock()
    llm.chat_with_structured_output = AsyncMock(
        return_value=AssertionResult(satisfied=True, reasoning="looks fine")
    )

    scenario = Scenario(
        id="s1",
        type=ScenarioType.POSITIVE,
        description="book a slot",
        turns=["hi"],
        assertions=[TranscriptAssertion(description="d", check="agent confirmed")],
    )
    transcript = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket")
    transcript.add_turn(Turn(user_message="hi", agent_response="", agent_failure=AgentFailure.NO_RESPONSE))

    state = {
        "current_batch_scenarios": [scenario],
        "transcripts": {"s1": transcript},
        "scenarios": [scenario],
    }
    out = await scorer_node(state, llm)
    verdict = out["verdicts"]["s1"]
    assert verdict.verdict == Verdict.FAILURE       # not ERROR, not SUCCESS
    assert "no_response" in verdict.reason
