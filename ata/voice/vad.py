"""Voice-activity detection for the voice channel.

"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

_INSTALL_HINT = (
    "The thick voice channel needs Silero VAD, which ships with the optional extra.\n"
    '    pip install "ata[pipecat]"'
)


class VadState(str, Enum):
    QUIET = "quiet"
    STARTING = "starting"
    SPEAKING = "speaking"
    STOPPING = "stopping"


@runtime_checkable
class VadDetector(Protocol):
    """Feed successive PCM chunks, get the current speech state. Stateful."""

    sample_rate: int

    async def analyze(self, pcm: bytes) -> VadState: ...


class SileroVadDetector:
    """``VadDetector`` backed by Pipecat's ``SileroVADAnalyzer`` (optional dep).

    Silero needs fixed-size frames (512 samples @16 kHz); arbitrary inbound chunks
    are buffered and processed frame by frame, returning the latest state.
    """

    def __init__(self, sample_rate: int = 16000, **vad_params):
        try:
            from pipecat.audio.vad.silero import SileroVADAnalyzer
            from pipecat.audio.vad.vad_analyzer import VADParams
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise ImportError(_INSTALL_HINT) from e

        self.sample_rate = sample_rate
        params = VADParams(**vad_params) if vad_params else None
        self._analyzer = SileroVADAnalyzer(sample_rate=sample_rate, params=params)
        self._frame_bytes = self._analyzer.num_frames_required() * 2  # 16-bit mono
        self._buf = bytearray()
        self._state = VadState.QUIET

    async def analyze(self, pcm: bytes) -> VadState:  # pragma: no cover - needs pipecat
        self._buf.extend(pcm)
        while len(self._buf) >= self._frame_bytes:
            frame = bytes(self._buf[: self._frame_bytes])
            del self._buf[: self._frame_bytes]
            pc_state = await self._analyzer.analyze_audio(frame)
            self._state = VadState(pc_state.name.lower())
        return self._state
