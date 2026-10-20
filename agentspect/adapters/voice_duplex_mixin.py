"""Duplex voice logic for voice adapters.


``send_turn`` runs **two concurrent activities** over the open connection, so both
directions stay live (which is what makes barge-in possible later):

- **Outbound** — consume the token stream, feed it to streaming TTS, and forward
  each audio frame to the agent as it is produced (a producer task fills a queue,
  the sender drains it). ``ATA_SPEECH_START`` is stamped when the *first frame is
  sent*, ``ATA_SPEECH_STOP`` after the last; then an ``end_of_speech`` frame tells
  the agent the turn is over.
- **Inbound** — a continuous receive loop (``_collect_agent_audio``) running in
  parallel: accumulate audio, run VAD, record TTFA / speech span / silence gaps and
  agent speech-start/stop events.

After both finish, STT transcribes the collected agent audio into the turn's text.

VAD is optional. With a ``VadDetector`` (``agentspect[pipecat]``) the inbound loop emits
agent speech-boundary events AND ends the turn on the speech→quiet edge — this is how a
raw carrier stream (telephony) gets endpointed, since it never sends an ``end_of_speech``
frame. Without a VAD, the turn ends only on an explicit ``end_of_speech`` frame or timeout
(a self-framing WS agent), and thin voice runs unchanged with no pipecat. Raw-PCM frames
(``{"format": "pcm"}``) are wrapped once into a WAV for the batch STT.
Content TTS/STT is still batch under the hood (``VoiceIO``); real streaming STT/TTS
drops in behind the same interface. Interruption/barge-in execution is deferred.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import json
import time
import wave
from collections.abc import AsyncIterator
from typing import NamedTuple

from websockets.exceptions import WebSocketException

from agentspect.models.transcript import AgentFailure, Turn, VoiceEvent, VoiceEventKind, VoiceMeta
from agentspect.voice.vad import VadState

_END_SIGNALS = {"end_of_speech", "end", "eos"}


class _AudioCapture(NamedTuple):
    audio: bytes
    ttfa_ms: int | None
    agent_speech_ms: int | None
    silence_gaps_ms: list[int]
    events: list[VoiceEvent]
    timed_out: bool
    is_pcm: bool = False  # True when frames were raw PCM (telephony) vs. WAV (mock)


class VoiceDuplexMixin:
    """Provides a streaming, concurrent ``send_turn`` for voice adapters.

    Expects on ``self``: ``voice_io``, ``timeout``, ``dead_air_gap_ms``; optional
    ``vad`` (``VadDetector | None``) and ``sample_rate``; and ``_connections``
    (from ``WebSocketConnectionMixin``).
    """

    vad = None
    sample_rate: int = 16000
    supports_token_stream = True  # the simulator streams LLM tokens into send_turn

    async def send_turn(self, session_id: str, message: str | AsyncIterator[str]) -> Turn:
        if session_id not in self._connections:
            text = message if isinstance(message, str) else ""
            return Turn(user_message=text, agent_response="", latency_ms=0,
                        error=f"Session not found: {session_id}")

        connection = self._connections[session_id]
        start = time.perf_counter()
        events: list[VoiceEvent] = []
        said: list[str] = []  # tokens actually sent
        out_frames: list[bytes] = []  # Outbound audio
        audio_q: asyncio.Queue[bytes | None] = asyncio.Queue()

        # Three concurrent activities over the open connection: inbound receive
        # (VAD + timing), continuous STT on that audio, and the outbound token→TTS
        # →send stream. All run at once so both directions stay live.
        inbound = asyncio.create_task(self._collect_agent_audio(connection, self.timeout, start, audio_q))
        stt = asyncio.create_task(self._consume_stt(audio_q))

        try:
            await self._stream_outbound(connection, message, events, said, start, out_frames)
            cap = await inbound
            agent_response, confidence = await stt
        except WebSocketException:
            await self._cancel(inbound, stt)
            return Turn(user_message="".join(said), agent_response="", latency_ms=self._ms(start),
                        agent_failure=AgentFailure.DISCONNECTED, voice_events=events)
        except Exception as e:
            await self._cancel(inbound, stt)
            return Turn(user_message="".join(said), agent_response="", latency_ms=self._ms(start),
                        error=f"Voice turn failed: {type(e).__name__}: {e}", voice_events=events)

        events.extend(cap.events)
        text = "".join(said)
        latency_ms = self._ms(start)
        ata_wav = self._frames_to_wav(out_frames)
        ata_b64 = base64.b64encode(ata_wav).decode("ascii") if ata_wav else None
        if not cap.audio:
            # Silence is only a failure in conversational context — see user_simulator.
            vm = self._voice_meta(cap)
            vm.ata_audio_wav_b64 = ata_b64
            return Turn(user_message=text, agent_response="", latency_ms=latency_ms,
                        voice=vm, voice_events=events)

        vm = self._voice_meta(cap, confidence)
        vm.ata_audio_wav_b64 = ata_b64
        return Turn(user_message=text, agent_response=agent_response, latency_ms=latency_ms,
                    voice=vm, voice_events=events)

    @staticmethod
    async def _cancel(*tasks) -> None:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _consume_stt(self, audio_q: asyncio.Queue) -> tuple[str, float | None]:
        """Continuously transcribe inbound audio from the queue, building the utterance."""
        async def chunks() -> AsyncIterator[bytes]:
            while True:
                chunk = await audio_q.get()
                if chunk is None:
                    break
                yield chunk

        texts: list[str] = []
        confidence: float | None = None
        async for result in self.voice_io.transcribe_stream(chunks()):
            if result.text:
                texts.append(result.text)
            if result.confidence is not None:
                confidence = result.confidence
        return " ".join(texts).strip(), confidence

    @staticmethod
    def _ms(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)

    @staticmethod
    def _as_token_stream(message: str | AsyncIterator[str]) -> AsyncIterator[str]:
        if not isinstance(message, str):
            return message

        async def _one() -> AsyncIterator[str]:
            yield message

        return _one()

    async def _stream_outbound(self, connection, message, events: list[VoiceEvent],
                               said: list[str], start: float, out_frames: list[bytes]) -> None:
        """Token stream → streaming TTS → audio frames → agent, as they are produced."""
        tokens = self._as_token_stream(message)
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def tee() -> AsyncIterator[str]:
            async for token in tokens:
                said.append(token)
                yield token

        async def produce() -> None:
            async for frame in self.voice_io.synthesize_stream(tee()):
                await queue.put(frame)
            await queue.put(None)  # sentinel

        producer = asyncio.create_task(produce())
        sent_any = False
        try:
            while True:
                frame = await queue.get()
                if frame is None:
                    break
                out_frames.append(frame)
                if not sent_any:
                    events.append(VoiceEvent(t_ms=self._ms(start), kind=VoiceEventKind.ATA_SPEECH_START, party="ata"))
                    sent_any = True
                await connection.send(json.dumps(
                    {"type": "audio", "data": base64.b64encode(frame).decode("ascii"), "format": "wav"}
                ))
            if sent_any:
                events.append(VoiceEvent(t_ms=self._ms(start), kind=VoiceEventKind.ATA_SPEECH_STOP, party="ata"))
            await connection.send(json.dumps({"type": "end_of_speech"}))  # ATA's turn is over
        finally:
            await producer

    async def _collect_agent_audio(
        self, connection, timeout: float, start: float, audio_q: asyncio.Queue | None = None
    ) -> _AudioCapture:
        """Continuous inbound loop: accumulate audio, run VAD, record timing + events.

        If ``audio_q`` is given, each audio chunk is also pushed to it (with a final
        ``None`` sentinel) so a concurrent STT consumer can transcribe as it arrives.
        """
        chunks: list[bytes] = []
        events: list[VoiceEvent] = []
        ttfa_ms: int | None = None
        first_t: float | None = None
        last_t: float | None = None
        gaps: list[int] = []
        vad_speaking = False
        is_pcm = False

        def finish(timed_out: bool) -> _AudioCapture:
            if vad_speaking:
                events.append(VoiceEvent(t_ms=self._ms(start), kind=VoiceEventKind.AGENT_SPEECH_STOP, party="agent"))
            if audio_q is not None:
                # Raw PCM is wrapped once here; per-frame WAV headers would corrupt the concat.
                if is_pcm and chunks:
                    audio_q.put_nowait(self._pcm_to_wav(b"".join(chunks)))
                audio_q.put_nowait(None)
            speech_ms = int((last_t - first_t) * 1000) if first_t is not None and last_t is not None else None
            return _AudioCapture(b"".join(chunks), ttfa_ms, speech_ms, gaps, events, timed_out, is_pcm)

        collect_started = time.perf_counter()
        while True:
            remaining = timeout - (time.perf_counter() - collect_started)
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
            if mtype in _END_SIGNALS:
                return finish(False)
            if mtype == "audio" and msg.get("data"):
                try:
                    chunk = base64.b64decode(msg["data"])
                except (ValueError, TypeError):
                    continue
                is_pcm = msg.get("format") == "pcm"  # raw-PCM stream (telephony) vs. WAV frames (mock)
                now = time.perf_counter()
                if ttfa_ms is None:
                    ttfa_ms = int((now - collect_started) * 1000)
                    first_t = now
                elif last_t is not None:
                    gap_ms = int((now - last_t) * 1000)
                    if gap_ms >= self.dead_air_gap_ms:
                        gaps.append(gap_ms)
                last_t = now
                chunks.append(chunk)
                if audio_q is not None and not is_pcm:
                    audio_q.put_nowait(chunk)  # PCM is wrapped once in finish()

                if self.vad is not None:
                    state = await self.vad.analyze(chunk)
                    if state == VadState.SPEAKING and not vad_speaking:
                        vad_speaking = True
                        events.append(VoiceEvent(t_ms=self._ms(start), kind=VoiceEventKind.AGENT_SPEECH_START, party="agent"))
                    elif state == VadState.QUIET and vad_speaking:
                        vad_speaking = False
                        events.append(VoiceEvent(t_ms=self._ms(start), kind=VoiceEventKind.AGENT_SPEECH_STOP, party="agent"))
                        return finish(False)  # VAD endpoint: agent went quiet → turn over

    def _pcm_to_wav(self, pcm: bytes) -> bytes:
        """Wrap raw PCM16 mono @ sample_rate into a WAV container for the batch STT."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm)
        return buf.getvalue()

    def _cap_wav(self, cap: _AudioCapture) -> bytes:
        """Captured audio as WAV bytes: wrap raw-PCM streams, pass WAV frames through."""
        return self._pcm_to_wav(cap.audio) if cap.is_pcm else cap.audio

    @staticmethod
    def _frames_to_wav(frames: list[bytes]) -> bytes:
        """Concatenate per-word WAV frames (outbound TTS) into one WAV.

        Each frame is its own WAV container, so a byte-concat would be invalid,
        decode each to PCM and re-wrap once, preserving the first frame's format.
        Non-WAV frames (e.g. a test fake) yield b\"\".
        """
        pcm = bytearray()
        params: tuple[int, int, int] | None = None
        for f in frames:
            try:
                with wave.open(io.BytesIO(f), "rb") as w:
                    params = params or (w.getnchannels(), w.getsampwidth(), w.getframerate())
                    pcm += w.readframes(w.getnframes())
            except (wave.Error, EOFError):
                continue
        if not pcm or params is None:
            return b""
        ch, width, rate = params
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(ch)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(bytes(pcm))
        return buf.getvalue()

    def _voice_meta(self, cap: _AudioCapture, stt_confidence: float | None = None) -> VoiceMeta:
        d = self.voice_io.defaults
        return VoiceMeta(
            time_to_first_audio_ms=cap.ttfa_ms,
            agent_speech_ms=cap.agent_speech_ms,
            silence_gaps_ms=list(cap.silence_gaps_ms),
            stt_confidence=stt_confidence,
            voice=d.voice,
            accent=d.accent,
            language=d.language,
            # inline the captured agent speech (WAV)
            audio_wav_b64=base64.b64encode(self._cap_wav(cap)).decode("ascii") if cap.audio else None,
        )
