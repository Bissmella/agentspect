from datetime import UTC, datetime
import json
import time

import websockets
from websockets.exceptions import WebSocketException

from backend.adapters.base import ProtocolAdapter
from backend.models.transcript import Turn


class WebSocketAdapter(ProtocolAdapter):
    def __init__(self, url: str, timeout: float = 30.0):
        super().__init__(url, timeout)
        self._connections: dict[str, websockets.WebSocketClientProtocol] = {}

    async def start_session(self, scenario_id: str) -> str:
        session_id = self.generate_session_id()

        try:
            connection = await websockets.connect(
                self.url,
                close_timeout=self.timeout,
                open_timeout=self.timeout,
            )
            self._connections[session_id] = connection
            return session_id
        except (WebSocketException, OSError) as e:
            raise ConnectionError(f"Failed to connect to WebSocket at {self.url}: {e}")

    async def send_turn(self, session_id: str, message: str) -> Turn:
        if session_id not in self._connections:
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=0,
                error=f"Session not found: {session_id}",
            )

        connection = self._connections[session_id]

        start_time = time.perf_counter()
        try:
            await connection.send(json.dumps({"message": message}))

            import asyncio

            response_raw = await asyncio.wait_for(connection.recv(), timeout=self.timeout)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            response_data = json.loads(response_raw)
            agent_response = response_data.get("response", response_data.get("message", ""))

            return Turn(
                user_message=message,
                agent_response=agent_response,
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
            )

        except TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
                error="TIMEOUT",
            )
        except WebSocketException as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
                error=f"WebSocket error: {str(e)}",
            )
        except json.JSONDecodeError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
                error=f"Invalid JSON response: {str(e)}",
            )

    async def end_session(self, session_id: str) -> None:
        if session_id in self._connections:
            connection = self._connections.pop(session_id)
            try:
                await connection.close()
            except WebSocketException:
                pass

    async def close(self) -> None:
        for session_id in list(self._connections.keys()):
            await self.end_session(session_id)


def create_adapter(protocol: str, url: str, timeout: float = 30.0) -> ProtocolAdapter:
    from backend.adapters.http_adapter import HTTPAdapter

    adapters = {
        "http": HTTPAdapter,
        "websocket": WebSocketAdapter,
    }

    if protocol not in adapters:
        raise ValueError(f"Unknown protocol: {protocol}. Must be one of: {list(adapters.keys())}")

    return adapters[protocol](url=url, timeout=timeout)
