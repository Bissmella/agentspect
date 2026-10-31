"""Tests for the thin voice channel: models, registry/BYO, VoiceIO, adapter, metric."""

import base64
import json

import pytest
from agentspect.adapters.base import ProtocolAdapter
from agentspect.adapters.voice_ws_adapter import VoiceWebSocketAdapter
from agentspect.adapters.ws_adapter import create_adapter
from agentspect.metrics.builtin import (
    AgentSpeechDurationMetric,
    DeadAirMetric,
    GreetingRateMetric,
    IntelligibilityMetric,
    NoResponseRateMetric,
    PrematureDisconnectMetric,
    TimeToFirstAudioMetric,
)
from agentspect.models.suite import Scenario, ScenarioType
from agentspect.models.transcript import AgentFailure, Transcript, Turn, VoiceMeta
from agentspect.models.yaml_input import STTSpec, TTSSpec, VoiceConfig
from agentspect.voice.client import (
    BatchSTTClient,
    STTClient,
    TranscriptionResult,
    TTSClient,
    VoiceIO,
    VoiceIODefaults,
)
from agentspect.voice.registry import create_voice_io, register_stt, register_tts

# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeSTT(BatchSTTClient):
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


def _pcm_frame(text: str) -> str:
    # A raw-PCM (telephony) frame — VAD runs on these; WS WAV frames use silence timing.
    return json.dumps({"type": "audio", "data": base64.b64encode(text.encode()).decode(), "format": "pcm"})


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
    from agentspect.voice.registry import stt_registry, tts_registry

    assert "openai" in stt_registry
    assert "openai" in tts_registry


# ── VoiceIO applies defaults ────────────────────────────────────────────────────

async def test_voice_io_synthesize_and_transcribe_roundtrip():
    vio = _fake_voice_io()
    audio = await vio.synthesize("book me a slot")
    result = await vio.transcribe(audio)
    assert result.text == "book me a slot"


async def test_synthesize_stream_batches_whole_reply_in_one_call():
    # Batch providers synthesize the whole reply in ONE call — per-sentence flushing
    # only multiplied HTTP calls (one connection each) with no benefit for batch TTS.
    vio = _fake_voice_io()  # FakeTTS returns the text bytes, so chunks decode to what was synth'd

    async def toks():
        for t in ["Hi ", "there. ", "Please ", "book ", "slot ", "A?"]:
            yield t

    chunks = [c.decode() async for c in vio.synthesize_stream(toks())]
    assert chunks == ["Hi there. Please book slot A?"]


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
    # the captured agent speech is inlined (b64) for the platform to offload
    assert turn.voice.audio_wav_b64 is not None
    assert base64.b64decode(turn.voice.audio_wav_b64) == b"You are booked."
    # the user's text was synthesized and sent as an audio frame
    assert json.loads(conn.sent[0])["type"] == "audio"
    await adapter.close()


def test_frames_to_wav_concatenates_per_word_wavs():
    import io
    import struct
    import wave

    from agentspect.adapters.voice_duplex_mixin import VoiceDuplexMixin

    def _wav(nframes, rate=16000):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(b"".join(struct.pack("<h", 1000) for _ in range(nframes)))
        return buf.getvalue()

    out = VoiceDuplexMixin._frames_to_wav([_wav(100), _wav(150)])
    with wave.open(io.BytesIO(out), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.getnframes() == 250  # per-word WAVs decoded + re-wrapped as one
    assert VoiceDuplexMixin._frames_to_wav([b"not a wav"]) == b""  # non-WAV → empty


async def test_adapter_silence_reports_blank_without_deciding_failure():
    # The agent goes silent. The adapter must NOT decide this is a failure (it lacks
    # conversational context); it returns a blank response and no agent_failure. (#5)
    conn = FakeConnection([_EOS], hang=True)
    adapter = _adapter_with(conn, timeout=0.2)
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello?")
    assert turn.error is None
    assert turn.agent_failure is None
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


async def test_adapter_empty_transcription_is_blank_not_unintelligible():
    # Audio that transcribes to whitespace is NOT claimed UNINTELLIGIBLE in batch STT
    # (unreliable); it is a blank response with no agent_failure. (#5)
    blank_audio = _audio_frame("   ")
    conn = FakeConnection([_EOS, blank_audio, _EOS])
    adapter = _adapter_with(conn)
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    assert turn.error is None
    assert turn.agent_failure is None
    assert not turn.agent_response.strip()
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


# ── Injectable connection (host-supplied, no network hop) ─────────────────────────

async def test_voice_callable_adapter_runs_duplex_over_host_connection():
    from agentspect.adapters import VoiceCallableAdapter

    conn = FakeConnection([_EOS, _audio_frame("You are booked."), _EOS])
    adapter = VoiceCallableAdapter(lambda: conn, voice_io=_fake_voice_io(), timeout=5.0)
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "I'd like slot A")
    assert turn.error is None
    assert turn.agent_response == "You are booked."
    assert conn.sent  # ATA's audio reached the host-supplied connection
    await adapter.close()
    assert conn.closed


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
    from agentspect.metrics.base import MetricContext

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
    from agentspect.metrics.base import MetricContext

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
    from agentspect.metrics.base import MetricContext

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
    """An agent behavioral failure (silence, hangup, unintelligible) is the agent's fault →
    FAILURE, not ERROR (ERROR is reserved for OUR-side faults). It fails the run even when
    the (mocked) assertion evaluator would say the check passed."""
    from unittest.mock import AsyncMock, MagicMock

    from agentspect.agents.scorer import AssertionResult, scorer_node
    from agentspect.models.suite import ScenarioType, TranscriptAssertion, Verdict

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


# ── Turn-position gate: blank/disconnect only fail when a reply was expected (#5, #6)

class _ScriptedAdapter(ProtocolAdapter):
    """Adapter that returns pre-scripted (agent_response, agent_failure) per turn."""

    def __init__(self, script):
        super().__init__(url="scripted://x")
        self._script = list(script)
        self._i = 0

    async def start_session(self, scenario_id: str) -> str:
        return "sess"

    async def send_turn(self, session_id: str, message: str) -> Turn:
        agent_response, agent_failure = self._script[self._i]
        self._i += 1
        return Turn(user_message=message, agent_response=agent_response, agent_failure=agent_failure)

    async def end_session(self, session_id: str) -> None:
        pass


def _mk_ws():
    from agentspect.models.world_state import WorldState

    return WorldState({"entities": [], "catalog": {}, "constraints": [], "context": {}})


async def test_blank_midconversation_is_no_response():
    from unittest.mock import AsyncMock, MagicMock

    from agentspect.agents.user_simulator import _run_single_scenario

    scenario = Scenario(id="s1", type=ScenarioType.POSITIVE, description="d", turns=["one", "two"])
    adapter = _ScriptedAdapter([("", None)])  # blank on turn 1 → should stop before turn 2
    llm = MagicMock()
    llm.chat = AsyncMock()

    transcript, verdict = await _run_single_scenario(scenario, _mk_ws(), adapter, llm)

    assert transcript.turns[0].agent_failure == AgentFailure.NO_RESPONSE
    assert verdict is None  # not ERROR — the scorer will score it
    llm.chat.assert_not_called()  # stopped before generating the second turn


async def test_blank_on_final_turn_is_not_a_failure():
    from unittest.mock import MagicMock

    from agentspect.agents.user_simulator import _run_single_scenario

    scenario = Scenario(id="s1", type=ScenarioType.POSITIVE, description="d", turns=["bye"])
    adapter = _ScriptedAdapter([("", None)])  # blank on the only/last turn → acceptable end

    transcript, verdict = await _run_single_scenario(scenario, _mk_ws(), adapter, MagicMock())

    assert transcript.turns[0].agent_failure is None
    assert verdict is None


async def test_disconnect_after_engagement_is_cleared_on_final_turn():
    from unittest.mock import AsyncMock, MagicMock

    from agentspect.agents.user_simulator import _run_single_scenario

    scenario = Scenario(id="s1", type=ScenarioType.POSITIVE, description="d", turns=["hi", "bye"])
    # Agent answers turn 1, then hangs up after goodbye — a graceful end-of-call.
    adapter = _ScriptedAdapter([("sure, how can I help?", None), ("", AgentFailure.DISCONNECTED)])
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=MagicMock(content="bye"))  # adapts the 2nd user turn

    transcript, verdict = await _run_single_scenario(scenario, _mk_ws(), adapter, llm)

    assert transcript.turns[-1].agent_failure is None  # gated away: agent had engaged
    assert verdict is None


async def test_disconnect_without_ever_engaging_is_not_cleared():
    from unittest.mock import MagicMock

    from agentspect.agents.user_simulator import _run_single_scenario

    scenario = Scenario(id="s1", type=ScenarioType.POSITIVE, description="d", turns=["bye"])
    # Agent never said anything and dropped — the timeout/disconnect bug, NOT a graceful hangup.
    adapter = _ScriptedAdapter([("", AgentFailure.DISCONNECTED)])

    transcript, verdict = await _run_single_scenario(scenario, _mk_ws(), adapter, MagicMock())

    assert transcript.turns[0].agent_failure == AgentFailure.DISCONNECTED  # kept → scorer makes it FAILURE
    assert verdict is None


# ── Stage A: streaming capture + observational metrics ─────────────────────────

async def test_adapter_captures_speech_span_and_gaps():
    # greeting (no audio), then three agent audio frames + end. With the dead-air
    # threshold at 0, every inter-frame gap is recorded → 2 gaps for 3 frames.
    af = _audio_frame("hi")
    conn = FakeConnection([_EOS, af, af, af, _EOS])
    adapter = VoiceWebSocketAdapter(
        url="wss://x", voice_io=_fake_voice_io(), timeout=5.0, dead_air_gap_ms=0
    )

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    assert turn.voice is not None
    assert turn.voice.agent_speech_ms is not None
    assert len(turn.voice.silence_gaps_ms) == 2
    await adapter.close()


def test_dead_air_metric():
    t = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket")
    t.add_turn(Turn(user_message="a", agent_response="b", voice=VoiceMeta(silence_gaps_ms=[600, 1200])))
    t.add_turn(Turn(user_message="c", agent_response="d", voice=VoiceMeta(silence_gaps_ms=[])))
    result = _run_metric(DeadAirMetric(), t)
    assert result.turns_with_gaps == 1
    assert result.total_gaps == 2
    assert result.max_gap_ms == 1200
    assert result.total_dead_air_ms == 1800


def test_agent_speech_duration_metric():
    t = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket")
    t.add_turn(Turn(user_message="a", agent_response="b", voice=VoiceMeta(agent_speech_ms=1000)))
    t.add_turn(Turn(user_message="c", agent_response="d", voice=VoiceMeta(agent_speech_ms=3000)))
    result = _run_metric(AgentSpeechDurationMetric(), t)
    assert result.turns == 2
    assert result.avg_ms == 2000.0
    assert result.total_ms == 4000


# ── Stage B (B3): timed voice-action scenario schema ───────────────────────────

def test_scenario_defaults_to_no_voice_actions():
    s = Scenario(id="s1", type=ScenarioType.POSITIVE, description="d", turns=["hi"])
    assert s.voice_actions == []


def test_valid_barge_in_action():
    from agentspect.models.suite import VoiceAction, VoiceActionType

    s = Scenario(
        id="s1", type=ScenarioType.POSITIVE, description="d", turns=["hi", "wait—"],
        voice_actions=[VoiceAction(type=VoiceActionType.BARGE_IN, turn_index=1, at_ms=500)],
    )
    assert s.voice_actions[0].at_ms == 500


def test_barge_in_requires_at_ms():
    from agentspect.models.suite import VoiceAction, VoiceActionType

    with pytest.raises(ValueError, match="at_ms"):
        VoiceAction(type=VoiceActionType.BARGE_IN, turn_index=0)


def test_dtmf_requires_digits():
    from agentspect.models.suite import VoiceAction, VoiceActionType

    with pytest.raises(ValueError, match="dtmf"):
        VoiceAction(type=VoiceActionType.DTMF, turn_index=0)


def test_voice_action_turn_index_out_of_range_rejected():
    from agentspect.models.suite import VoiceAction, VoiceActionType

    with pytest.raises(ValueError, match="out of range"):
        Scenario(
            id="s1", type=ScenarioType.POSITIVE, description="d", turns=["only one turn"],
            voice_actions=[VoiceAction(type=VoiceActionType.SILENCE, turn_index=3)],
        )


# ── Stage B2: event-timeline model ─────────────────────────────────────────────

def test_transcript_voice_events_default_empty():
    t = Transcript(scenario_id="s1", session_id="x", protocol="http")
    assert t.voice_events == []


def test_transcript_voice_events_flattens_turns():
    from agentspect.models.transcript import VoiceEvent, VoiceEventKind

    ev = VoiceEvent(t_ms=120, kind=VoiceEventKind.AGENT_SPEECH_START, party="agent")
    t = Transcript(scenario_id="s1", session_id="x", protocol="voicewebsocket")
    t.add_turn(Turn(user_message="a", agent_response="b", voice_events=[ev]))
    # scenario-level timeline is derived from per-turn events
    assert len(t.voice_events) == 1
    assert t.voice_events[0].kind == VoiceEventKind.AGENT_SPEECH_START


# ── Stage B1: VAD guard + duplex session ───────────────────────────────────────

def test_silero_vad_detector_requires_extra():
    # Only meaningful without the extra: it asserts an actionable ImportError. In a
    # voice-enabled env (pipecat installed) that path can't be exercised, so skip.
    import importlib.util

    if importlib.util.find_spec("pipecat") is not None:
        pytest.skip("pipecat installed — the missing-extra path can't be exercised here")

    from agentspect.voice.vad import SileroVadDetector

    with pytest.raises(ImportError, match=r"agentspect\[pipecat\]"):
        SileroVadDetector()


class _ScriptedVad:
    """Fake VadDetector returning a scripted state per analyze() call."""

    sample_rate = 16000

    def __init__(self, states):
        from agentspect.voice.vad import VadState

        self._states = list(states)
        self._quiet = VadState.QUIET

    async def analyze(self, pcm: bytes):
        return self._states.pop(0) if self._states else self._quiet


def _kinds(events):
    return [e.kind.value for e in events]


async def test_adapter_with_vad_emits_agent_speech_events():
    # With a VAD, the receive path marks the agent's speech boundaries on the turn.
    from agentspect.voice.vad import VadState

    conn = FakeConnection([_EOS, _audio_frame("a"), _audio_frame("b"), _EOS])
    adapter = VoiceWebSocketAdapter(
        url="wss://x", voice_io=_fake_voice_io(), timeout=1.0,
        vad=_ScriptedVad([VadState.SPEAKING, VadState.QUIET]),
    )

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    kinds = _kinds(turn.voice_events)
    assert kinds[0] == "ata_speech_start"  # ATA's own send is always marked
    assert "agent_speech_start" in kinds and "agent_speech_stop" in kinds
    await adapter.close()


async def test_vad_ends_turn_without_end_of_speech_frame():
    # A raw carrier (PCM) stream never sends end_of_speech. The VAD speech→quiet edge
    # must end the turn on its own (no trailing _EOS; VAD runs only on PCM frames).
    from agentspect.voice.vad import VadState

    conn = FakeConnection([_EOS, _pcm_frame("a"), _pcm_frame("b")])  # no final _EOS
    adapter = VoiceWebSocketAdapter(
        url="wss://x", voice_io=_fake_voice_io(), timeout=5.0,
        vad=_ScriptedVad([VadState.SPEAKING, VadState.QUIET]),
    )

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    assert turn.error is None
    assert turn.agent_response                       # transcribed; ended on VAD quiet, not timeout
    assert "agent_speech_stop" in _kinds(turn.voice_events)
    await adapter.close()


async def test_raw_pcm_frames_are_wrapped_to_wav_for_stt():
    # Telephony frames carry raw PCM; the engine must hand the STT one valid WAV,
    # not a concatenation of per-frame headers.
    import io
    import wave

    from agentspect.voice.vad import VadState

    class CapturingSTT(BatchSTTClient):
        def __init__(self, model=None, api_key=None):
            self.last_audio: bytes | None = None

        async def transcribe(self, audio, *, language=None) -> TranscriptionResult:
            self.last_audio = audio
            return TranscriptionResult(text="ok", confidence=0.5, language=language)

    def _pcm_frame(pcm: bytes) -> str:
        return json.dumps({"type": "audio", "data": base64.b64encode(pcm).decode(), "format": "pcm"})

    vio = VoiceIO(stt=CapturingSTT(), tts=FakeTTS(),
                  defaults=VoiceIODefaults(voice="alloy", language="en"))
    frame = _pcm_frame(b"\x00\x01" * 160)  # 160 samples of PCM16
    conn = FakeConnection([_EOS, frame, frame])
    adapter = VoiceWebSocketAdapter(
        url="wss://x", voice_io=vio, timeout=5.0,
        vad=_ScriptedVad([VadState.SPEAKING, VadState.QUIET]),
    )

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    assert turn.error is None
    assert vio.stt.last_audio is not None and vio.stt.last_audio[:4] == b"RIFF"
    with wave.open(io.BytesIO(vio.stt.last_audio), "rb") as w:
        assert w.getnchannels() == 1 and w.getframerate() == 16000


async def test_multi_wav_frames_are_rejoined_into_one_valid_wav_for_stt():
    # A real agent streams its reply as MANY WAV frames. The engine must decode + rejoin
    # them into one valid WAV for the STT, never byte-concatenate the containers (which
    # makes the STT read only the first header and garble the rest → "stuck" words).
    import io
    import struct
    import wave

    def _wav(nframes: int) -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(struct.pack("<h", 1000) for _ in range(nframes)))
        return json.dumps({"type": "audio", "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"})

    class CapturingSTT(BatchSTTClient):
        def __init__(self, model=None, api_key=None):
            self.last_audio: bytes | None = None

        async def transcribe(self, audio, *, language=None) -> TranscriptionResult:
            self.last_audio = audio
            return TranscriptionResult(text="ok", confidence=0.5, language=language)

    vio = VoiceIO(stt=CapturingSTT(), tts=FakeTTS(),
                  defaults=VoiceIODefaults(voice="alloy", language="en"))
    conn = FakeConnection([_EOS, _wav(100), _wav(150)], hang=True)
    adapter = VoiceWebSocketAdapter(url="wss://x", voice_io=vio, timeout=5.0, silence_ms=100)

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    assert turn.error is None
    assert vio.stt.last_audio is not None and vio.stt.last_audio[:4] == b"RIFF"
    with wave.open(io.BytesIO(vio.stt.last_audio), "rb") as w:
        assert w.getnframes() == 250  # two frames rejoined, not a corrupt byte-concat
    await adapter.close()


async def test_agent_speech_ms_is_audio_duration_not_frame_arrival_span():
    # Talk time must be the audio's own duration — a single-frame reply has ~0 arrival
    # span but is clearly non-zero speech. Regression for "agent talk time: 0 ms".
    import io
    import struct
    import wave

    def _wav(nframes: int) -> str:  # nframes @16k → nframes/16 ms of audio
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(b"".join(struct.pack("<h", 1000) for _ in range(nframes)))
        return json.dumps({"type": "audio", "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"})

    conn = FakeConnection([_EOS, _wav(16000)], hang=True)  # one 1000 ms frame
    adapter = VoiceWebSocketAdapter(url="wss://x", voice_io=_fake_voice_io(), timeout=5.0, silence_ms=100)

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    assert turn.voice.agent_speech_ms == 1000  # the audio's duration, not ~0 arrival span
    await adapter.close()


async def test_pcm_vad_waits_for_speech_before_endpointing():
    # Telephony: a continuous PCM stream opens with silence (VAD QUIET) before the agent
    # speaks. The turn must NOT end on that leading silence — only on the speech→quiet
    # edge — else a live agent is wrongly judged silent. Regression for the VAD gate.
    from agentspect.voice.vad import VadState

    conn = FakeConnection(
        [_EOS, _pcm_frame("s"), _pcm_frame("s"), _pcm_frame("hello"), _pcm_frame("x")],
        hang=True,
    )
    adapter = VoiceWebSocketAdapter(
        url="wss://x", voice_io=_fake_voice_io(), timeout=5.0,
        vad=_ScriptedVad([VadState.QUIET, VadState.QUIET, VadState.SPEAKING, VadState.QUIET]),
    )

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    assert turn.error is None
    assert turn.agent_response                       # captured speech, not the leading silence
    kinds = _kinds(turn.voice_events)
    assert "agent_speech_start" in kinds and "agent_speech_stop" in kinds
    await adapter.close()


async def test_pcm_all_silence_is_a_silent_turn():
    # Continuous PCM silence (agent never speaks): frames keep arriving, so endpointing
    # can't rely on a recv gap — the time-based AWAIT budget expires → a blank turn with
    # no false speech capture.
    from agentspect.voice.vad import VadState

    conn = FakeConnection([_EOS, _pcm_frame("s"), _pcm_frame("s")], hang=True)
    adapter = VoiceWebSocketAdapter(
        url="wss://x", voice_io=_fake_voice_io(), timeout=5.0,
        vad=_ScriptedVad([VadState.QUIET]), response_timeout=0.2,
    )

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    assert turn.error is None
    assert turn.agent_response == ""                 # never spoke → silent, no false capture
    assert "agent_speech_start" not in _kinds(turn.voice_events)
    await adapter.close()


async def test_adapter_records_agent_speech_events_without_vad():
    # No VAD (WS frames / thin voice) still yields a complete state timeline: the
    # adapter marks when the agent started and stopped speaking (first frame → silence),
    # so speaking/listening is tracked even without pipecat.
    conn = FakeConnection([_EOS, _audio_frame("a"), _EOS])
    adapter = _adapter_with(conn)
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    kinds = _kinds(turn.voice_events)
    assert "ata_speech_start" in kinds
    assert "agent_speech_start" in kinds
    assert "agent_speech_stop" in kinds
    await adapter.close()


async def test_adapter_endpoints_on_silence_without_end_of_speech():
    # WS agent speaks, then just stops — no end_of_speech frame. The turn must end on
    # the inter-frame silence gap, and ATA must never send end_of_speech either.
    conn = FakeConnection([_EOS, _audio_frame("You are booked.")], hang=True)
    adapter = VoiceWebSocketAdapter(
        url="wss://x", voice_io=_fake_voice_io(), timeout=5.0, silence_ms=100
    )

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "I'd like slot A")
    assert turn.error is None
    assert turn.agent_response == "You are booked."     # endpointed on silence, no _EOS
    assert all(json.loads(m).get("type") != "end_of_speech" for m in conn.sent)
    await adapter.close()


async def test_listening_is_not_cut_off_by_a_short_timeout():
    # The agent may talk longer than `timeout`; only silence ends its turn, never a
    # stopwatch. Regression: a prior hard cap truncated the agent mid-speech.
    import asyncio

    class _PacedConnection:
        def __init__(self, frames, gap):
            self._frames = list(frames)
            self._gap = gap
            self.sent = []

        async def send(self, m):
            self.sent.append(m)

        async def recv(self):
            if self._frames:
                await asyncio.sleep(self._gap)
                return self._frames.pop(0)
            await asyncio.sleep(3600)  # then go quiet → LISTENING inter-frame gap ends it

        async def close(self):
            pass

    frames = [_EOS] + [_audio_frame(c) for c in "abcdefgh"]  # _EOS = greeting
    conn = _PacedConnection(frames, gap=0.05)                # 8 speech frames over ~0.4s
    adapter = VoiceWebSocketAdapter(
        url="wss://x", voice_io=_fake_voice_io(), timeout=0.2, silence_ms=120
    )

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")           # greeting consumes _EOS
    turn = await adapter.send_turn(session_id, "go")
    assert turn.error is None
    assert turn.agent_response == "abcdefgh"                  # full speech, not truncated at 0.2s
    await adapter.close()


async def test_adapter_outbound_synth_error_is_platform_error_not_agent_failure():
    # Our own TTS/stream raising is a PLATFORM failure (Turn.error) — never charged to
    # the agent as a failure. It must surface promptly, not hang.
    import asyncio

    class BoomTTS(FakeTTS):
        async def synthesize(self, text, *, voice=None, language=None, accent=None) -> bytes:
            raise RuntimeError("tts provider down")

    vio = VoiceIO(stt=FakeSTT(), tts=BoomTTS(),
                  defaults=VoiceIODefaults(voice="alloy", language="en", accent=None))
    conn = FakeConnection([_EOS], hang=True)
    adapter = VoiceWebSocketAdapter(url="wss://x", voice_io=vio, timeout=2.0, silence_ms=100)

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await asyncio.wait_for(adapter.send_turn(session_id, "hello"), timeout=5.0)
    assert turn.error is not None          # platform failure surfaced
    assert turn.agent_failure is None      # not blamed on the agent
    await adapter.close()


async def test_send_turn_accepts_a_token_stream():
    # send_turn takes str OR an async iterator of tokens (the streaming shape).
    async def tokens():
        for t in ["he", "llo"]:
            yield t

    conn = FakeConnection([_EOS, _audio_frame("ok"), _EOS])
    adapter = _adapter_with(conn)
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, tokens())
    assert turn.user_message == "hello"  # tokens materialized
    await adapter.close()


# ── Simulator → adapter token streaming ────────────────────────────────────────

class _FakeStreamLLM:
    async def chat(self, messages, **kwargs):
        from agentspect.llm.client import LLMResponse

        return LLMResponse(content="full message")

    async def chat_stream(self, messages, **kwargs):
        for tok in ["Hel", "lo"]:
            yield tok


class _RecordingAdapter(ProtocolAdapter):
    supports_token_stream = True

    def __init__(self):
        super().__init__(url="rec://x")
        self.message_types: list[str] = []

    async def start_session(self, scenario_id: str) -> str:
        return "sess"

    async def send_turn(self, session_id, message):
        if isinstance(message, str):
            self.message_types.append("str")
            text = message
        else:
            self.message_types.append("stream")
            text = "".join([t async for t in message])
        return Turn(user_message=text, agent_response="ok")

    async def end_session(self, session_id: str) -> None:
        pass


async def test_user_simulator_streams_tokens_to_streaming_adapter():
    from agentspect.agents.user_simulator import _run_single_scenario
    from agentspect.models.world_state import WorldState

    scenario = Scenario(id="s1", type=ScenarioType.POSITIVE, description="d", turns=["hi", "again"])
    adapter = _RecordingAdapter()
    ws = WorldState({"entities": [], "catalog": {}, "constraints": [], "context": {}})

    transcript, _ = await _run_single_scenario(scenario, ws, adapter, _FakeStreamLLM())

    # turn 0 is a plain string; the adapted turn 1 is streamed as LLM tokens
    assert adapter.message_types == ["str", "stream"]
    # and the streamed turn's text was assembled from the tokens, not the batch chat()
    assert transcript.turns[1].user_message == "Hello"


# ── Streaming STT: inbound transcription via transcribe_stream ─────────────────

async def test_transcribe_stream_default_is_batch_backed():
    vio = _fake_voice_io()

    async def chunks():
        for part in [b"book ", b"me a slot"]:
            yield part

    results = [r async for r in vio.transcribe_stream(chunks())]
    assert len(results) == 1  # batch-backed default: one final result
    assert results[0].text == "book me a slot"


async def test_transcribe_stream_uses_provider_streaming_override():
    # A provider that implements its own streaming ASR (e.g. Gradium's WS) owns it in the
    # connector; VoiceIO just delegates, so the adapter stays provider-agnostic.
    from agentspect.voice.client import STTClient, TranscriptionResult

    class _StreamingSTT(STTClient):
        async def transcribe(self, audio, *, language=None):  # batch path unused here
            raise AssertionError("streaming provider should not fall back to batch")

        async def transcribe_stream(self, audio_chunks, *, language=None):
            async for _ in audio_chunks:
                pass
            yield TranscriptionResult(text="streamed", confidence=0.8, language=language)

    vio = VoiceIO(_StreamingSTT(), FakeTTS(), VoiceIODefaults(language="en"))

    async def chunks():
        yield b"audio"

    results = [r async for r in vio.transcribe_stream(chunks())]
    assert [r.text for r in results] == ["streamed"]
    assert results[0].language == "en"  # VoiceIO passes its default language through


async def test_adapter_builds_response_from_streaming_stt_partials():
    from agentspect.voice.client import TranscriptionResult

    class _StreamingVoiceIO(VoiceIO):
        async def transcribe_stream(self, audio_chunks):
            async for _ in audio_chunks:  # drain inbound audio
                pass
            yield TranscriptionResult(text="booked", confidence=0.7)
            yield TranscriptionResult(text="now", confidence=0.9)

    conn = FakeConnection([_EOS, _audio_frame("x"), _EOS])
    vio = _StreamingVoiceIO(stt=FakeSTT(), tts=FakeTTS(),
                            defaults=VoiceIODefaults(voice="alloy", language="en"))
    adapter = VoiceWebSocketAdapter(url="wss://x", voice_io=vio, timeout=1.0)

    async def fake_open(url):
        return conn

    adapter._open_connection = fake_open
    session_id = await adapter.start_session("s1")
    turn = await adapter.send_turn(session_id, "hello")
    # the agent response is assembled from the streamed partial transcripts
    assert turn.agent_response == "booked now"
    assert turn.voice.stt_confidence == 0.9  # last partial's confidence
    await adapter.close()
