"""Voice I/O abstraction

Thin voice needs only batch request/responseو synthesize text to audio, and
transcribe audio to text.
Streaming / realtime / VAD are deferred to the Pipecat bridge.

STT and TTS are separate abstractions so a run can mix providers (e.g. one for
speech-to-text, another for text-to-speech) or use a provider that only does one.
``VoiceIO`` pairs a chosen STT and TTS client and applies the run's voice/language/
accent defaults, recording which were used so the adapter can stamp ``VoiceMeta``.
"""

from __future__ import annotations

import io
import wave
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel


class TranscriptionResult(BaseModel):
    text: str
    confidence: float | None = None
    language: str | None = None


class STTClient(ABC):
    """Speech-to-text. Implementations transcribe raw audio bytes.

    Both methods are abstract: a provider states explicitly how it streams. One with no
    streaming ASR extends ``BatchSTTClient`` to opt into the buffer-once fallback.
    """

    @abstractmethod
    async def transcribe(self, audio: bytes, *, language: str | None = None) -> TranscriptionResult:
        ...

    @abstractmethod
    def transcribe_stream(
        self, audio_chunks: AsyncIterator[bytes], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        """Yield final transcripts as audio streams in (immediate; no end-of-turn batch wait)."""
        ...


class BatchSTTClient(STTClient):
    """STT base for providers with no streaming ASR (e.g. OpenAI Whisper).

    The only option there is to accumulate the audio and transcribe once — so the
    batch lives here, opt-in, not as a default on every STT.
    """

    async def transcribe_stream(
        self, audio_chunks: AsyncIterator[bytes], *, language: str | None = None
    ) -> AsyncIterator[TranscriptionResult]:
        # Frames arrive as standalone WAV containers (the adapter streams one per chunk).
        # A naive byte-concat would glue WAV headers mid-stream and corrupt the audio, so
        # decode each to PCM and re-wrap once; non-WAV frames raw-concat as a fallback.
        pcm = bytearray()
        params: tuple[int, int, int] | None = None
        async for chunk in audio_chunks:
            try:
                with wave.open(io.BytesIO(chunk), "rb") as w:
                    params = params or (w.getnchannels(), w.getsampwidth(), w.getframerate())
                    pcm += w.readframes(w.getnframes())
            except (wave.Error, EOFError):
                pcm += chunk
        if not pcm:
            return
        if params is None:
            audio = bytes(pcm)
        else:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(params[0])
                w.setsampwidth(params[1])
                w.setframerate(params[2])
                w.writeframes(bytes(pcm))
            audio = buf.getvalue()
        yield await self.transcribe(audio, language=language)


class TTSClient(ABC):
    """Text-to-speech. Implementations return raw audio bytes."""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        accent: str | None = None,
    ) -> bytes:
        ...

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        accent: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Yield audio frames as they are produced. Batch default: one blob.

        A provider whose API streams the audio (first bytes long before the last)
        overrides this to forward chunks as they arrive — the downstream STT then
        transcribes in realtime instead of waiting for the whole clip.
        """
        yield await self.synthesize(text, voice=voice, language=language, accent=accent)


class VoiceIODefaults(BaseModel):
    """Run-level voice defaults, passed through opaquely to the provider."""

    voice: str | None = None
    language: str | None = None
    accent: str | None = None


class VoiceIO:
    """A resolved STT + TTS pair plus the run's defaults.

    The voice adapter uses this rather than the individual clients. It records the
    voice/language/accent it spoke with so ``VoiceMeta`` reflects the actual call.
    """

    def __init__(self, stt: STTClient, tts: TTSClient, defaults: VoiceIODefaults | None = None):
        self.stt = stt
        self.tts = tts
        self.defaults = defaults or VoiceIODefaults()

    async def transcribe(self, audio: bytes) -> TranscriptionResult:
        """Transcribe one audio blob through the provider's STREAMING path.
        """
        async def _one() -> AsyncIterator[bytes]:
            yield audio

        texts: list[str] = []
        confidence: float | None = None
        language: str | None = None
        async for r in self.stt.transcribe_stream(_one(), language=self.defaults.language):
            if r.text:
                texts.append(r.text)
            if r.confidence is not None:
                confidence = r.confidence
            language = r.language or language
        return TranscriptionResult(text=" ".join(texts).strip(), confidence=confidence, language=language)

    async def synthesize(self, text: str) -> bytes:
        return await self.tts.synthesize(
            text,
            voice=self.defaults.voice,
            language=self.defaults.language,
            accent=self.defaults.accent,
        )

    async def synthesize_stream(self, tokens: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Accumulate the reply text, then stream audio frames from the TTS as produced.

        A streaming TTS yields many frames (first audio fast); a batch one yields a single
        blob. ``tokens`` is fully consumed, so callers that tee it (to record what was
        said) are unaffected.
        """
        buffer = ""
        async for token in tokens:
            buffer += token
        text = buffer.strip()
        if not text:
            return
        async for frame in self.tts.synthesize_stream(
            text, voice=self.defaults.voice, language=self.defaults.language, accent=self.defaults.accent
        ):
            yield frame

    async def transcribe_stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        """Delegate to the STT provider's own streaming path (it owns its ASR + VAD).

        Providers without a streaming ASR inherit ``STTClient.transcribe_stream``'s
        batch default, so this stays provider-agnostic either way.
        """
        async for result in self.stt.transcribe_stream(
            audio_chunks, language=self.defaults.language
        ):
            yield result
