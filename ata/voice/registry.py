"""Registries for STT and TTS providers.

Providers register themselves (usually via ``@register_stt`` / ``@register_tts``)
so a run can select them by name from the YAML ``voice`` block. This is the same
pattern used for metrics and adapters: ATA ships one reference provider (OpenAI)
and anyone can bring their own in a few lines::

    from ata.voice.registry import register_stt
    from ata.voice.client import STTClient, TranscriptionResult

    @register_stt("deepgram")
    class DeepgramSTT(STTClient):
        def __init__(self, model=None, api_key=None): ...
        async def transcribe(self, audio, *, language=None): ...

The long tail of providers is also reachable through the optional Pipecat bridge
(``ata.voice.bridges.pipecat``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from ata.voice.client import VoiceIO, VoiceIODefaults

_C = TypeVar("_C", bound=type)


class _Registry:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, type] = {}

    def register(self, name: str, *, replace: bool = False) -> Callable[[_C], _C]:
        def decorator(cls: _C) -> _C:
            if name in self._entries and not replace:
                raise ValueError(
                    f"{self._kind} provider '{name}' is already registered; "
                    "pass replace=True to override"
                )
            self._entries[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._entries:
            raise KeyError(
                f"No {self._kind} provider registered under '{name}'. Known: {self.names()}"
            )
        return self._entries[name]

    def names(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries


stt_registry = _Registry("STT")
tts_registry = _Registry("TTS")


def register_stt(name: str, *, replace: bool = False) -> Callable[[_C], _C]:
    return stt_registry.register(name, replace=replace)


def register_tts(name: str, *, replace: bool = False) -> Callable[[_C], _C]:
    return tts_registry.register(name, replace=replace)


def create_voice_io(
    *,
    stt_provider: str,
    tts_provider: str,
    stt_model: str | None = None,
    tts_model: str | None = None,
    voice: str | None = None,
    language: str | None = None,
    accent: str | None = None,
    stt_api_key: str | None = None,
    tts_api_key: str | None = None,
) -> VoiceIO:
    """Resolve STT/TTS providers from the registries and build a ``VoiceIO``.

    Importing ``ata.voice`` registers the built-in reference (OpenAI) providers.
    """

    stt_cls = stt_registry.get(stt_provider)
    tts_cls = tts_registry.get(tts_provider)
    stt = stt_cls(model=stt_model, api_key=stt_api_key)
    tts = tts_cls(model=tts_model, api_key=tts_api_key)
    defaults = VoiceIODefaults(voice=voice, language=language, accent=accent)
    return VoiceIO(stt=stt, tts=tts, defaults=defaults)
