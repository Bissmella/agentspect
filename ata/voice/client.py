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
