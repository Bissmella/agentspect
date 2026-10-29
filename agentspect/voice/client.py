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

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from pydantic import BaseModel


class TranscriptionResult(BaseModel):
    text: str
    confidence: float | None = None
    language: str | None = None


class STTClient(ABC):
    """Speech-to-text. Implementations transcribe raw audio bytes."""

    @abstractmethod
    async def transcribe(self, audio: bytes, *, language: str | None = None) -> TranscriptionResult:
        ...


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
        return await self.stt.transcribe(audio, language=self.defaults.language)

    async def synthesize(self, text: str) -> bytes:
        return await self.tts.synthesize(
            text,
            voice=self.defaults.voice,
            language=self.defaults.language,
            accent=self.defaults.accent,
        )

    async def synthesize_stream(self, tokens: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Batch-backed default: accumulate the whole reply and synthesize it in ONE call.

        This default is only ever used by batch (REST) providers — a real streaming TTS
        (Pipecat) overrides the method. The old per-sentence flush turned one reply into
        several HTTP calls (a new connection each) and hammered batch providers'
        concurrency caps, with no real benefit: incremental first-audio only helps a true
        streaming TTS. ``tokens`` is still fully consumed, so callers that tee it (to record
        what was said) are unaffected.
        """
        buffer = ""
        async for token in tokens:
            buffer += token
        text = buffer.strip()
        if text:
            yield await self.synthesize(text)

    async def transcribe_stream(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[TranscriptionResult]:
        """Stream STT: transcribe inbound audio as it arrives, yielding final results.

        This batch-backed default accumulates the chunks and transcribes once (one
        final result). A real streaming STT (Pipecat) overrides this to yield final
        transcripts as they finalize, so the caller builds the agent's utterance
        without waiting for the agent to finish speaking.
        """
        buffer = bytearray()
        async for chunk in audio_chunks:
            buffer.extend(chunk)
        if buffer:
            yield await self.transcribe(bytes(buffer))
