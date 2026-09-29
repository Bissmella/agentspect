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

from websockets.exceptions import WebSocketException

from ata.adapters.base import ProtocolAdapter
from ata.adapters.ws_base import WebSocketConnectionMixin
from ata.models.transcript import AgentFailure, Transcript, Turn, VoiceMeta
from ata.voice.client import VoiceIO

_END_SIGNALS = {"end_of_speech", "end", "eos"}


class VoiceWebSocketAdapter(WebSocketConnectionMixin, ProtocolAdapter):
    def __init__(
        self,
        url: str,
        voice_io: VoiceIO,
        timeout: float = 30.0,
        greeting_timeout: float | None = None,
    ):
        super().__init__(url, timeout)
        self.voice_io = voice_io
        # Agents that don't greet shouldn't stall the whole timeout.
        self.greeting_timeout = greeting_timeout if greeting_timeout is not None else min(timeout, 5.0)
        self._openings: dict[str, tuple[str, VoiceMeta] | None] = {}

    async def start_session(self, scenario_id: str) -> str:
        session_id, connection = await self._connect_session("voice WebSocket")

        # Capture an opening greeting if the agent speaks first (best-effort).
        try:
            audio, ttfa_ms, _ = await self._collect_agent_audio(connection, self.greeting_timeout)
            if audio:
                result = await self.voice_io.transcribe(audio)
                self._openings[session_id] = (
                    result.text,
                    self._voice_meta(ttfa_ms, result.confidence),
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

            audio_in, ttfa_ms, _timed_out = await self._collect_agent_audio(connection, self.timeout)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            
            if not audio_in:
                return Turn(
                    user_message=message,
                    agent_response="",
                    latency_ms=latency_ms,
                    voice=self._voice_meta(ttfa_ms, None),
                )

            result = await self.voice_io.transcribe(audio_in)
            return Turn(
                user_message=message,
                agent_response=result.text,
                latency_ms=latency_ms,
                voice=self._voice_meta(ttfa_ms, result.confidence),
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

    async def _collect_agent_audio(
        self, connection, timeout: float
    ) -> tuple[bytes, int | None, bool]:
        """Collect agent audio frames until end-of-speech or a silence timeout.

        Returns (audio_bytes, time_to_first_audio_ms, timed_out).
        """
        chunks: list[bytes] = []
        ttfa_ms: int | None = None
        started = time.perf_counter()

        while True:
            remaining = timeout - (time.perf_counter() - started)
            if remaining <= 0:
                return b"".join(chunks), ttfa_ms, True
            try:
                raw = await asyncio.wait_for(connection.recv(), timeout=remaining)
            except TimeoutError:
                return b"".join(chunks), ttfa_ms, True

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue

            mtype = msg.get("type")
            if mtype == "audio" and msg.get("data"):
                if ttfa_ms is None:
                    ttfa_ms = int((time.perf_counter() - started) * 1000)
                try:
                    chunks.append(base64.b64decode(msg["data"]))
                except (ValueError, TypeError):
                    continue
            elif mtype in _END_SIGNALS:
                return b"".join(chunks), ttfa_ms, False

    def _voice_meta(self, ttfa_ms: int | None, stt_confidence: float | None) -> VoiceMeta:
        d = self.voice_io.defaults
        return VoiceMeta(
            time_to_first_audio_ms=ttfa_ms,
            stt_confidence=stt_confidence,
            voice=d.voice,
            accent=d.accent,
            language=d.language,
        )

    async def end_session(self, session_id: str) -> None:
        self._openings.pop(session_id, None)
        await super().end_session(session_id)
