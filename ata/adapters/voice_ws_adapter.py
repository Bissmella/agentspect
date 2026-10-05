"""Voice-over-WebSocket adapter.

Wire convention (ATA voice-WS protocol) — JSON text frames::

    ATA  -> agent:  {"type": "audio", "data": "<base64>", "format": "wav"}
    agent-> ATA:    {"type": "audio", "data": "<base64>"}   (one or more)
    agent-> ATA:    {"type": "end_of_speech"}               (turn complete)

"""

from __future__ import annotations

import time

from ata.adapters.base import ProtocolAdapter
from ata.adapters.voice_duplex_mixin import VoiceDuplexMixin
from ata.adapters.ws_base import WebSocketConnectionMixin
from ata.models.transcript import Transcript, VoiceMeta
from ata.voice.client import VoiceIO
from ata.voice.vad import VadDetector

_DEFAULT_DEAD_AIR_GAP_MS = 500  # inter-frame gap above which mid-response silence counts as dead air


class VoiceWebSocketAdapter(VoiceDuplexMixin, WebSocketConnectionMixin, ProtocolAdapter):
    def __init__(
        self,
        url: str,
        voice_io: VoiceIO,
        timeout: float = 30.0,
        greeting_timeout: float | None = None,
        dead_air_gap_ms: int = _DEFAULT_DEAD_AIR_GAP_MS,
        vad: VadDetector | None = None,
    ):
        super().__init__(url, timeout)
        self.voice_io = voice_io
        # Agents that don't greet shouldn't stall the whole timeout.
        self.greeting_timeout = greeting_timeout if greeting_timeout is not None else min(timeout, 5.0)
        self.dead_air_gap_ms = dead_air_gap_ms
        self.vad = vad
        if vad is not None:
            self.sample_rate = vad.sample_rate
        self._openings: dict[str, tuple[str, VoiceMeta] | None] = {}

    async def start_session(self, scenario_id: str) -> str:
        session_id, connection = await self._connect_session("voice WebSocket")

        # Capture an opening greeting if the agent speaks first (best-effort).
        try:
            cap = await self._collect_agent_audio(connection, self.greeting_timeout, time.perf_counter())
            if cap.audio:
                result = await self.voice_io.transcribe(cap.audio)
                self._openings[session_id] = (result.text, self._voice_meta(cap, result.confidence))
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

    async def end_session(self, session_id: str) -> None:
        self._openings.pop(session_id, None)
        await super().end_session(session_id)
