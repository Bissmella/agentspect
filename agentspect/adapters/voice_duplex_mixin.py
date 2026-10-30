"""Duplex voice logic for voice adapters.

A turn is driven by an explicit conversation state machine (``CallState``) — the model a
real call uses, where **silence** drives the turn-taking and nobody announces end-of-speech:

    SPEAKING   ATA streams its utterance (token stream → TTS → frames). It does NOT
               send end_of_speech; it simply stops sending.
    AWAITING   ATA has finished; it waits up to ``response_timeout`` for the agent to
               *start* speaking. If the agent never does, the turn is a silent
               (no-response) one for the simulator to judge. This is the only clock on
               the agent, and it bounds *starting* — never how long the agent may talk.
    LISTENING  the agent is speaking; ATA captures audio until the speech→silence edge
               ends the turn: a VAD quiet edge on a continuous PCM stream (telephony),
               or an inter-frame silence gap of ``silence_ms`` on a frame protocol (WS).
               An incoming end_of_speech is honored as an optional early end, never
               required. There is deliberately **no cap on the agent's speaking time**.
    ENDED      turn complete; STT transcribes the captured audio into the turn's text.

Inbound receive + STT run concurrently with SPEAKING so the agent may begin replying
before ATA finishes; Stage 1 consumes the result after speaking (barge-in is deferred).
VAD is optional (``agentspect[pipecat]``): it endpoints a continuous PCM stream, while WS
WAV frames (a header per frame) use the inter-frame silence gap instead. Each WAV frame is
streamed to the STT as it arrives, so transcription overlaps the agent still speaking; the
TTS likewise yields frames as produced. A raw-PCM (telephony) stream is still transcribed
as one blob at end-of-turn. Batch providers yield a single blob behind the same interface.
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
from enum import Enum
from typing import NamedTuple

from websockets.exceptions import WebSocketException

from agentspect.models.transcript import AgentFailure, Turn, VoiceEvent, VoiceEventKind, VoiceMeta
from agentspect.voice.vad import VadState

_END_SIGNALS = {"end_of_speech", "end", "eos"}


class CallState(str, Enum):
    """Where a voice turn is in its lifecycle. Silence drives the transitions."""

    SPEAKING = "speaking"    # ATA is streaming its own utterance
    AWAITING = "awaiting"    # ATA done; waiting for the agent to start speaking
    LISTENING = "listening"  # agent is speaking; capturing until the silence edge
    ENDED = "ended"          # turn complete


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
    ``vad`` (``VadDetector | None``), ``sample_rate``, ``silence_ms``,
    ``response_timeout``; and ``_connections`` (from ``WebSocketConnectionMixin``).
    """

    vad = None
    sample_rate: int = 16000
    silence_ms: int = 700           # quiet gap (speech→silence) that ends a turn
    response_timeout: float = 15.0  # how long to wait for the agent to start replying
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

        # Inbound receive + continuous STT start immediately (the agent may begin
        # replying before we finish speaking).
        inbound = asyncio.create_task(self._collect_agent_audio(connection, self.response_timeout, start, audio_q))
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
            # Always leave a sentinel so the drain loop can't block forever; any
            # synth/LLM-stream error then surfaces via ``await producer`` below —
            # a platform failure, reported as such, never as the agent's.
            try:
                async for frame in self.voice_io.synthesize_stream(tee()):
                    await queue.put(frame)
            finally:
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
        finally:
            await producer  # re-raises a producer error → caught by send_turn → ERROR

    async def _collect_agent_audio(
        self, connection, first_audio_timeout: float, start: float, audio_q: asyncio.Queue | None = None
    ) -> _AudioCapture:
        """Run the AWAITING → LISTENING → ENDED states of a turn and capture the agent's audio.

        AWAITING waits up to ``first_audio_timeout`` for the agent to *start*; no speech by
        then → a silent (no-response) turn. LISTENING captures until the speech→silence edge
        (VAD quiet on PCM, an inter-frame ``silence_ms`` gap on WS, or an optional
        end_of_speech). There is deliberately **no cap on how long the agent may speak** —
        only silence ends its turn. If ``audio_q`` is given, the captured audio is pushed as a
        single valid WAV (then a ``None`` sentinel) for the STT consumer.
        """
        chunks: list[bytes] = []
        events: list[VoiceEvent] = []
        ttfa_ms: int | None = None
        first_t: float | None = None
        last_t: float | None = None
        gaps: list[int] = []
        is_pcm = False
        silence_s = max(self.silence_ms / 1000.0, 0.05)
        collect_started = time.perf_counter()
        state = CallState.AWAITING

        def finish() -> _AudioCapture:
            # Reached speech → the agent spoke; still AWAITING → it never did (silent turn).
            if state is CallState.LISTENING:
                events.append(VoiceEvent(t_ms=self._ms(start), kind=VoiceEventKind.AGENT_SPEECH_STOP, party="agent"))
            if not chunks:
                wav = b""
            elif is_pcm:
                wav = self._pcm_to_wav(b"".join(chunks))
            else:
                wav = self._join_wav_frames(chunks)
            if audio_q is not None:
                if is_pcm and wav:  # PCM wasn't streamed per-frame → hand the STT the blob
                    audio_q.put_nowait(wav)
                audio_q.put_nowait(None)
            # Talk time is the captured audio's own duration, not the frame-arrival span
            speech_ms = self._wav_duration_ms(wav)
            if speech_ms is None and first_t is not None and last_t is not None:
                speech_ms = int((last_t - first_t) * 1000)
            return _AudioCapture(
                wav, ttfa_ms, speech_ms, gaps, events,
                timed_out=state is not CallState.LISTENING, is_pcm=is_pcm,
            )

        def begin_listening(now: float) -> None:
            nonlocal state, ttfa_ms, first_t
            state = CallState.LISTENING
            ttfa_ms = int((now - collect_started) * 1000)
            first_t = now
            events.append(VoiceEvent(t_ms=self._ms(start), kind=VoiceEventKind.AGENT_SPEECH_START, party="agent"))

        def note_gap(now: float) -> None:
            if last_t is not None:
                gap_ms = int((now - last_t) * 1000)
                if gap_ms >= self.dead_air_gap_ms:
                    gaps.append(gap_ms)

        def feed_stt(chunk: bytes) -> None:
            # Stream each WAV frame to the STT as it arrives
            if audio_q is not None and not is_pcm:
                audio_q.put_nowait(chunk)

        while True:
            if state is CallState.AWAITING:
                elapsed = time.perf_counter() - collect_started
                if elapsed >= first_audio_timeout:  # agent never started → silent turn
                    return finish()
                window = max(first_audio_timeout - elapsed, 0.01)
            else:  # LISTENING — a quiet gap this long is the agent's speech→silence edge
                window = silence_s
            try:
                raw = await asyncio.wait_for(connection.recv(), timeout=window)
            except TimeoutError:
                return finish()  # AWAITING → agent silent; LISTENING → it finished speaking

            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            mtype = msg.get("type")
            if mtype in _END_SIGNALS:  # optional early-end hint from a self-framing agent
                return finish()
            if mtype != "audio" or not msg.get("data"):
                continue
            try:
                chunk = base64.b64decode(msg["data"])
            except (ValueError, TypeError):
                continue
            is_pcm = msg.get("format") == "pcm"  # raw-PCM stream (telephony) vs. WAV frames (mock)
            now = time.perf_counter()

            # PCM / continuous stream: VAD marks speech boundaries. Leading silence keeps us
            # AWAITING; a quiet edge once LISTENING is the agent's speech→silence endpoint.
            if self.vad is not None and is_pcm:
                quiet = (await self.vad.analyze(chunk)) == VadState.QUIET
                if state is CallState.AWAITING:
                    if quiet:
                        continue  # not speaking yet — stay in AWAITING
                    begin_listening(now)
                elif quiet:
                    last_t = now
                    chunks.append(chunk)
                    feed_stt(chunk)
                    return finish()  # speech → quiet edge: turn over
                else:
                    note_gap(now)
                last_t = now
                chunks.append(chunk)
                feed_stt(chunk)
                continue

            # WS / WAV frames (no VAD): a frame arrives only when the agent speaks, so the
            # first frame begins LISTENING and an inter-frame silence gap (above) ends it.
            if state is CallState.AWAITING:
                begin_listening(now)
            else:
                note_gap(now)
            last_t = now
            chunks.append(chunk)  # kept for the inlined report audio; STT gets it live below
            feed_stt(chunk)

    def _pcm_to_wav(self, pcm: bytes) -> bytes:
        """Wrap raw PCM16 mono @ sample_rate into a WAV container for the batch STT."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm)
        return buf.getvalue()

    def _join_wav_frames(self, frames: list[bytes]) -> bytes:
        """One valid WAV from per-frame WAV containers; raw-concat fallback for
        non-WAV frames (a raw stream, or a test fake that isn't a real WAV)."""
        return self._frames_to_wav(frames) or b"".join(frames)

    @staticmethod
    def _wav_duration_ms(wav: bytes) -> int | None:
        """Audio duration of a WAV in ms, or None if the bytes aren't a readable WAV."""
        if not wav:
            return None
        try:
            with wave.open(io.BytesIO(wav), "rb") as w:
                rate = w.getframerate()
                return int(w.getnframes() / rate * 1000) if rate else None
        except (wave.Error, EOFError):
            return None

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
            # inline the captured agent speech (already one valid WAV from finish())
            audio_wav_b64=base64.b64encode(cap.audio).decode("ascii") if cap.audio else None,
        )
