"""In-process voice adapter.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from agentspect.adapters.voice_ws_adapter import VoiceWebSocketAdapter

ConnectionFactory = Callable[[], Any | Awaitable[Any]]


class VoiceCallableAdapter(VoiceWebSocketAdapter):
    def __init__(self, connect: ConnectionFactory, *args, **kwargs):
        if not callable(connect):
            raise TypeError("VoiceCallableAdapter requires a connection factory callable")
        super().__init__("voice-callable://in-process", *args, **kwargs)
        self._connect = connect

    async def _open_connection(self, url: str):
        conn = self._connect()
        return await conn if inspect.isawaitable(conn) else conn
