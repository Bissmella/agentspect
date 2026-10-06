from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentFailure(str, Enum):
    """A behavioral failure of the agent under test, observed over the transport.
    agents' falut not framework error.
    Names are transport-neutral so a future telephony adapter reuses them as-is.
    """

    NO_RESPONSE = "no_response"          # agent stayed silent / returned nothing
    DISCONNECTED = "disconnected"        # agent dropped the conversation early
    UNINTELLIGIBLE = "unintelligible"    # reserved: voice speech below an STT-confidence threshold (needs a provider that reports confidence)
    CALLER_REJECTED = "caller_rejected"  # reserved: agent refused the caller (telephony, platform-side)


class VoiceMeta(BaseModel):
    """Per-turn voice telemetry, captured by voice adapters.

    Every field is optional: text adapters never set it.
    Used for voice-based metrics.
    """

    time_to_first_audio_ms: int | None = None  # start of agent speech, not round-trip
    agent_speech_ms: int | None = None
    user_speech_ms: int | None = None
    silence_gaps_ms: list[int] = Field(default_factory=list)
    interrupted: bool | None = None  # reserved for barge-in orchestration
    dtmf: str | None = None
    stt_confidence: float | None = None
    voice: str | None = None  # the TTS voice ATA spoke with
    accent: str | None = None
    language: str | None = None
    audio_ref: str | None = None  # blob key for recorded audio, if stored


class VoiceEventKind(str, Enum):
    """Timestamped events on a duplex voice conversation (Stage B, thick)."""

    AGENT_SPEECH_START = "agent_speech_start"  # from VAD on the agent's audio
    AGENT_SPEECH_STOP = "agent_speech_stop"
    ATA_SPEECH_START = "ata_speech_start"      # ATA began sending its own audio
    ATA_SPEECH_STOP = "ata_speech_stop"
    DTMF_SENT = "dtmf_sent"
    SILENCE_WINDOW = "silence_window"          # ATA deliberately sent nothing and waited


class VoiceEvent(BaseModel):
    t_ms: int  # milliseconds since the duplex session started
    kind: VoiceEventKind
    party: str  # "agent" | "ata"
    detail: str | None = None


class Turn(BaseModel):
    user_message: str
    agent_response: str
    timestamp: datetime = Field(default_factory=_utcnow)
    latency_ms: int = 0
    error: str | None = None  # framework/transport fault (ATA's side) -> ERROR verdict
    agent_failure: AgentFailure | None = None  # agent's behavioral fault -> scored as a failed run
    voice: VoiceMeta | None = None
    voice_events: list[VoiceEvent] = Field(default_factory=list)  # duplex timeline for this turn


class Transcript(BaseModel):
    scenario_id: str
    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    protocol: str
    # Voice agents speak first; captured here so turn counts stay clean.
    opening_utterance: str | None = None
    opening_voice: VoiceMeta | None = None

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)

    def finalize(self) -> None:
        self.ended_at = datetime.now(UTC)

    @property
    def has_error(self) -> bool:
        return any(turn.error is not None for turn in self.turns)

    @property
    def last_error(self) -> str | None:
        for turn in reversed(self.turns):
            if turn.error:
                return turn.error
        return None

    @property
    def agent_failure(self) -> AgentFailure | None:
        """The last behavioral failure the agent exhibited, if any."""
        for turn in reversed(self.turns):
            if turn.agent_failure is not None:
                return turn.agent_failure
        return None

    @property
    def voice_events(self) -> list[VoiceEvent]:
        """The duplex voice timeline for the whole conversation (flattened turns)."""
        return [event for turn in self.turns for event in turn.voice_events]
