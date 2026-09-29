"""OpenAI STT/TTS

Both sides: Whisper for transcription
and the TTS models for synthesis. Keys come from the environment via ``settings``,
never from the YAML — the same rule as the LLM client.
"""

from __future__ import annotations

import io

from ata.config import settings
from ata.voice.client import STTClient, TranscriptionResult, TTSClient
from ata.voice.registry import register_stt, register_tts

_DEFAULT_STT_MODEL = "whisper-1"
_DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
_DEFAULT_VOICE = "alloy"


@register_stt("openai")
class OpenAISTT(STTClient):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        import openai

        self.model = model or _DEFAULT_STT_MODEL
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def transcribe(self, audio: bytes, *, language: str | None = None) -> TranscriptionResult:
        buffer = io.BytesIO(audio)
        buffer.name = "audio.wav"  # the SDK infers format from the filename
        kwargs: dict = {"model": self.model, "file": buffer}
        if language:
            kwargs["language"] = language
        response = await self.client.audio.transcriptions.create(**kwargs)
        return TranscriptionResult(
            text=getattr(response, "text", "") or "",
            confidence=None,  # Whisper's HTTP API does not expose a scalar confidence
            language=getattr(response, "language", None) or language,
        )


@register_tts("openai")
class OpenAITTS(TTSClient):
    def __init__(self, model: str | None = None, api_key: str | None = None):
        import openai

        self.model = model or _DEFAULT_TTS_MODEL
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.openai_api_key)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        accent: str | None = None,
    ) -> bytes:
        # language/accent are steered through the voice choice for OpenAI; passed
        # through opaquely rather than normalized.
        response = await self.client.audio.speech.create(
            model=self.model,
            voice=voice or _DEFAULT_VOICE,
            input=text,
        )
        return response.read()
