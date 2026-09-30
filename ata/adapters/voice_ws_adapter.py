"""Voice-over-WebSocket adapter.

Keeps the lockstep ``send_turn(text) -> Turn`` contract of the text adapters, but
each turn round-trips through audio: the user text is synthesized to speech (TTS),
streamed to the agent, the agent's spoken reply is collected and transcribed back
to text (STT). 
Wire convention (ATA voice-WS protocol) — JSON text frames::

    ATA  -> agent:  {"type": "audio", "data": "<base64>", "format": "wav"}
    agent-> ATA:    {"type": "audio", "data": "<base64>"}   (one or more)
    agent-> ATA:    {"type": "end_of_speech"}               (turn complete)

The agent may emit an opening ``audio ... end_of_speech`` burst right after connect
(agents speak first); it is captured as the transcript's ``opening_utterance``. A
target that speaks a different protocol is bridged to this convention by a small
shim or by the Pipecat bridge. Endpointing prefers the ``end_of_speech`` signal;
absent it, a silence timeout ends the turn.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import NamedTuple

from websockets.exceptions import WebSocketException

from ata.adapters.base import ProtocolAdapter
from ata.adapters.ws_base import WebSocketConnectionMixin
from ata.models.transcript import AgentFailure, Transcript, Turn, VoiceMeta
from ata.voice.client import VoiceIO

_END_SIGNALS = {"end_of_speech", "end", "eos"}
_DEFAULT_DEAD_AIR_GAP_MS = 500  # inter-frame gap above which mid-response silence counts as dead air


class _AudioCapture(NamedTuple):
    """What one agent-audio collection observed (Stage A streaming telemetry)."""

    audio: bytes
    ttfa_ms: int | None          # time to first audio frame
    agent_speech_ms: int | None  # first-to-last audio-frame span
    silence_gaps_ms: list[int]   # mid-response gaps above the dead-air threshold
    timed_out: bool


class VoiceWebSocketAdapter(WebSocketConnectionMixin, ProtocolAdapter):
    def __init__(
        self,
        url: str,
        voice_io: VoiceIO,
        timeout: float = 30.0,
        greeting_timeout: float | None = None,
        dead_air_gap_ms: int = _DEFAULT_DEAD_AIR_GAP_MS,
    ):
        super().__init__(url, timeout)
        self.voice_io = voice_io
        # Agents that don't greet shouldn't stall the whole timeout.
        self.greeting_timeout = greeting_timeout if greeting_timeout is not None else min(timeout, 5.0)
        self.dead_air_gap_ms = dead_air_gap_ms
        self._openings: dict[str, tuple[str, VoiceMeta] | None] = {}

    async def start_session(self, scenario_id: str) -> str:
        session_id, connection = await self._connect_session("voice WebSocket")

        # Capture an opening greeting if the agent speaks first (best-effort).
        try:
            cap = await self._collect_agent_audio(connection, self.greeting_timeout)
            if cap.audio:
                result = await self.voice_io.transcribe(cap.audio)
                self._openings[session_id] = (
                    result.text,
                    self._voice_meta(cap, result.confidence),
                )
            else:
                self._openings[session_id] = None
        except Exception:
            # A missing/failed greeting is not an error; the turn loop proceeds.
            self._openings[session_id] = None

        return session_id

    def create_transcript(self, scenario_id: str, session_id: str, protocol: str) -> Transcript:
        transcript = super().create_transcript(scenario_id, session_id, protocol)
        opening = self._openings.get(session_id)
        if opening is not None:
            transcript.opening_utterance = opening[0]
            transcript.opening_voice = opening[1]
        return transcript

    async def send_turn(self, session_id: str, message: str) -> Turn:
        if session_id not in self._connections:
            return Turn(
                user_message=message,
                agent_response="",
                latency_ms=0,
                error=f"Session not found: {session_id}",
            )

        connection = self._connections[session_id]
        start_time = time.perf_counter()

        try:
            audio_out = await self.voice_io.synthesize(message)
            await connection.send(
                json.dumps(
                    {
                        "type": "audio",
                        "data": base64.b64encode(audio_out).decode("ascii"),
                        "format": "wav",
                    }
                )
            )

            cap = await self._collect_agent_audio(connection, self.timeout)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            if not cap.audio:
                return Turn(
                    user_message=message,
                    agent_response="",
                    latency_ms=latency_ms,
                    voice=self._voice_meta(cap, None),
                )

            result = await self.voice_io.transcribe(cap.audio)
            return Turn(
                user_message=message,
                agent_response=result.text,
                latency_ms=latency_ms,
                voice=self._voice_meta(cap, result.confidence),
            )

        except WebSocketException:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Turn(
                user_message=message,
                agent_response="",
                latency_ms=latency_ms,
                agent_failure=AgentFailure.DISCONNECTED,
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Turn(
                user_message=message,
                agent_response="",
                latency_ms=latency_ms,
                error=f"Voice turn failed: {type(e).__name__}: {e}",
            )

    async def _collect_agent_audio(self, connection, timeout: float) -> _AudioCapture:
        """Collect agent audio frames until end-of-speech or a silence timeout.

        Timestamps frame arrivals to capture time-to-first-audio, the speech span,
        and mid-response silence gaps (Stage A streaming telemetry).
        """
        chunks: list[bytes] = []
        ttfa_ms: int | None = None
        first_audio_t: float | None = None
        last_audio_t: float | None = None
        gaps_ms: list[int] = []
        started = time.perf_counter()

        def finish(timed_out: bool) -> _AudioCapture:
            speech_ms = None
            if first_audio_t is not None and last_audio_t is not None:
                speech_ms = int((last_audio_t - first_audio_t) * 1000)
            return _AudioCapture(b"".join(chunks), ttfa_ms, speech_ms, gaps_ms, timed_out)

        while True:
            remaining = timeout - (time.perf_counter() - started)
            if remaining <= 0:
                return finish(True)
            try:
                raw = await asyncio.wait_for(connection.recv(), timeout=remaining)
            except TimeoutError:
                return finish(True)

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            mtype = msg.get("type")
            if mtype == "audio" and msg.get("data"):
                try:
                    chunk = base64.b64decode(msg["data"])
                except (ValueError, TypeError):
                    continue
                now = time.perf_counter()
                if ttfa_ms is None:
                    ttfa_ms = int((now - started) * 1000)
                    first_audio_t = now
                elif last_audio_t is not None:
                    gap_ms = int((now - last_audio_t) * 1000)
                    if gap_ms >= self.dead_air_gap_ms:
                        gaps_ms.append(gap_ms)
                last_audio_t = now
                chunks.append(chunk)
            elif mtype in _END_SIGNALS:
                return finish(False)

    def _voice_meta(self, cap: _AudioCapture, stt_confidence: float | None) -> VoiceMeta:
        d = self.voice_io.defaults
        return VoiceMeta(
            time_to_first_audio_ms=cap.ttfa_ms,
            agent_speech_ms=cap.agent_speech_ms,
            silence_gaps_ms=list(cap.silence_gaps_ms),
            stt_confidence=stt_confidence,
            voice=d.voice,
            accent=d.accent,
            language=d.language,
        )

    async def end_session(self, session_id: str) -> None:
        self._openings.pop(session_id, None)
        await super().end_session(session_id)
