"""In-process adapter: drive a Python callable as the agent under test.

This is still black-box / outside-in — ATA only ever sees what the callable
returns for a given message. It does not instrument, decorate, or inspect the
agent's internals. It is simply the HTTP/WebSocket contract without a network
hop, so a plain Python agent can be tested with zero server plumbing.

The callable may be sync or async, and may accept either just the message or
the message plus the conversation history::

    def agent(message: str) -> str: ...
    async def agent(message: str, history: list[dict]) -> str: ...

``history`` is the list of prior turns in this session, each a
``{"user": str, "agent": str}`` dict, oldest first.
"""

from collections.abc import Callable
import inspect
import time
from datetime import UTC, datetime
from typing import Any

from agentspect.adapters.base import ProtocolAdapter
from agentspect.models.transcript import Turn

AgentCallable = Callable[..., Any]


def _accepts_history(fn: AgentCallable) -> bool:
    """True if the callable takes a second positional arg (the history)."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return False
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    has_var_positional = any(
        p.kind == p.VAR_POSITIONAL for p in sig.parameters.values()
    )
    return has_var_positional or len(positional) >= 2


class CallableAdapter(ProtocolAdapter):
    def __init__(self, agent: AgentCallable, timeout: float = 30.0):
        super().__init__(url="callable://in-process", timeout=timeout)
        if not callable(agent):
            raise TypeError("CallableAdapter requires a callable agent")
        self._agent = agent
        self._wants_history = _accepts_history(agent)
        self._histories: dict[str, list[dict[str, str]]] = {}

    async def start_session(self, scenario_id: str) -> str:
        session_id = self.generate_session_id()
        self._histories[session_id] = []
        return session_id

    async def send_turn(self, session_id: str, message: str) -> Turn:
        if session_id not in self._histories:
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=0,
                error=f"Session not found: {session_id}",
            )

        history = self._histories[session_id]
        args: tuple[Any, ...] = (message, list(history)) if self._wants_history else (message,)

        start_time = time.perf_counter()
        try:
            result = self._agent(*args)
            if inspect.isawaitable(result):
                result = await result
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            agent_response = "" if result is None else str(result)
            history.append({"user": message, "agent": agent_response})

            return Turn(
                user_message=message,
                agent_response=agent_response,
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
                error=f"Callable raised {type(e).__name__}: {e}",
            )

    async def end_session(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    async def close(self) -> None:
        self._histories.clear()
