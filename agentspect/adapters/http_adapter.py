from datetime import UTC, datetime
import time

import httpx

from agentspect.adapters.base import ProtocolAdapter
from agentspect.models.transcript import Turn


class HTTPAdapter(ProtocolAdapter):
    def __init__(self, url: str, timeout: float = 30.0):
        super().__init__(url, timeout)
        self._clients: dict[str, httpx.AsyncClient] = {}

    async def start_session(self, scenario_id: str) -> str:
        session_id = self.generate_session_id()
        self._clients[session_id] = httpx.AsyncClient(timeout=self.timeout)
        return session_id

    async def send_turn(self, session_id: str, message: str) -> Turn:
        if session_id not in self._clients:
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=0,
                error=f"Session not found: {session_id}",
            )

        client = self._clients[session_id]

        start_time = time.perf_counter()
        try:
            response = await client.post(
                self.url,
                json={"message": message, "session_id": session_id},
            )
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            client.cookies.update(response.cookies)

            if response.status_code != 200:
                return Turn(
                    user_message=message,
                    agent_response="",
                    timestamp=datetime.now(UTC),
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                )

            data = response.json()
            agent_response = data.get("response", data.get("message", ""))

            return Turn(
                user_message=message,
                agent_response=agent_response,
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
            )

        except httpx.TimeoutException:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
                error="TIMEOUT",
            )
        except httpx.RequestError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
                error=f"Connection error: {str(e)}",
            )

    async def end_session(self, session_id: str) -> None:
        if session_id in self._clients:
            client = self._clients.pop(session_id)
            if not client.is_closed:
                await client.aclose()

    async def close(self) -> None:
        for client in self._clients.values():
            if not client.is_closed:
                await client.aclose()
        self._clients.clear()
