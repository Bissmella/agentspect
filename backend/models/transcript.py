from datetime import UTC, datetime

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Turn(BaseModel):
    user_message: str
    agent_response: str
    timestamp: datetime = Field(default_factory=_utcnow)
    latency_ms: int = 0
    error: str | None = None


class Transcript(BaseModel):
    scenario_id: str
    session_id: str
    turns: list[Turn] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    protocol: str

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
