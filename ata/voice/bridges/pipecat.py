"""Optional Pipecat bridge — reach the long tail of STT/TTS providers.

Pipecat (https://github.com/pipecat-ai/pipecat) maintains service plugins for
dozens of STT/TTS providers (Deepgram, ElevenLabs, Cartesia, Azure, ...). Exposing
a single bridge that adapts a Pipecat service to ATA's ``STTClient`` / ``TTSClient``
interface, so the whole ecosystem is reachable and maintained upstream.

This is an OPTIONAL extra — Pipecat is not a core dependency.
"""

from __future__ import annotations

from ata.voice.client import STTClient, TranscriptionResult, TTSClient

_INSTALL_HINT = (
    "The Pipecat bridge requires the optional extra. Install it with:\n"
    '    pip install "ata[pipecat]"'
)


def _require_pipecat():
    try:
        import pipecat  # noqa: F401
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from e


class PipecatSTT(STTClient):
    """Adapt a Pipecat STT service to ATA's batch ``STTClient``.

    ``service`` is a constructed Pipecat STT service instance.
    """

    def __init__(self, service=None, model: str | None = None, api_key: str | None = None):
        _require_pipecat()
        if service is None:
            raise NotImplementedError(
                "PipecatSTT currently requires a pre-constructed Pipecat STT service "
                "passed as `service=`."
            )
        self._service = service

    async def transcribe(self, audio: bytes, *, language: str | None = None) -> TranscriptionResult:  # pragma: no cover
        raise NotImplementedError(
            "Pipecat batch transcription glue is not wired yet (scaffold). "
            "Track it in docs/voice-plan.md."
        )


class PipecatTTS(TTSClient):
    """Adapt a Pipecat TTS service to ATA's batch ``TTSClient``."""

    def __init__(self, service=None, model: str | None = None, api_key: str | None = None):
        _require_pipecat()
        if service is None:
            raise NotImplementedError(
                "PipecatTTS currently requires a pre-constructed Pipecat TTS service "
                "passed as `service=`. Auto-construction from provider name is a "
                "planned follow-up (see docs/voice-plan.md)."
            )
        self._service = service

    async def synthesize(  # pragma: no cover
        self,
        text: str,
        *,
        voice: str | None = None,
        language: str | None = None,
        accent: str | None = None,
    ) -> bytes:
        raise NotImplementedError(
            "Pipecat batch synthesis glue is not wired yet (scaffold). "
            "Track it in docs/voice-plan.md."
        )
