
# Register the built-in reference providers (OpenAI STT + TTS).
from ata.voice import providers as _providers  # noqa: F401
from ata.voice.client import (
    STTClient,
    TranscriptionResult,
    TTSClient,
    VoiceIO,
    VoiceIODefaults,
)
from ata.voice.registry import (
    create_voice_io,
    register_stt,
    register_tts,
    stt_registry,
    tts_registry,
)

__all__ = [
    "STTClient",
    "TTSClient",
    "TranscriptionResult",
    "VoiceIO",
    "VoiceIODefaults",
    "create_voice_io",
    "register_stt",
    "register_tts",
    "stt_registry",
    "tts_registry",
]
