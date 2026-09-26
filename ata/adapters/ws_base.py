"""Shared WebSocket connection lifecycle for WS-based adapters.
"""

from __future__ import annotations

import websockets
from websockets.exceptions import WebSocketException


class WebSocketConnectionMixin:
    # Provided by ProtocolAdapter via the MRO: self.url, self.timeout,
    # self.generate_session_id().

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._connections: dict[str, object] = {}

    async def _open_connection(self, url: str):
        """Open a WebSocket connection."""
        return await websockets.connect(url, open_timeout=self.timeout, close_timeout=self.timeout)

    async def _connect_session(self, error_label: str) -> tuple[str, object]:
        """Open a connection, register it under a new session id, and return both.

        Raises ``ConnectionError`` on failure so the caller surfaces an ERROR
        verdict rather than crashing.
        """
        session_id = self.generate_session_id()
        try:
            connection = await self._open_connection(self.url)
        except (WebSocketException, OSError) as e:
            raise ConnectionError(f"Failed to connect to {error_label} at {self.url}: {e}")
        self._connections[session_id] = connection
        return session_id, connection

    async def end_session(self, session_id: str) -> None:
        connection = self._connections.pop(session_id, None)
        if connection is not None:
            try:
                await connection.close()
            except WebSocketException:
                pass

    async def close(self) -> None:
        for session_id in list(self._connections.keys()):
            await self.end_session(session_id)
