from abc import ABC, abstractmethod
from datetime import UTC, datetime
import uuid

from agentspect.models.transcript import Transcript, Turn


class ProtocolAdapter(ABC):
    # Cap on scenarios the executor runs in parallel against this adapter; None = unlimited.
    # Voice sets this low: the STT/TTS provider has an account-wide concurrent-session cap,
    # so parallel scenarios would thrash it and inflate the latency we report.
    max_concurrency: int | None = None

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url
        self.timeout = timeout

    @abstractmethod
    async def start_session(self, scenario_id: str) -> str:
        pass

    @abstractmethod
    async def send_turn(self, session_id: str, message: str) -> Turn:
        pass

    @abstractmethod
    async def end_session(self, session_id: str) -> None:
        pass

    async def close(self) -> None:
        pass

    def create_transcript(self, scenario_id: str, session_id: str, protocol: str) -> Transcript:
        return Transcript(
            scenario_id=scenario_id,
            session_id=session_id,
            protocol=protocol,
            turns=[],
            started_at=datetime.now(UTC),
        )

    @staticmethod
    def generate_session_id() -> str:
        return str(uuid.uuid4())
