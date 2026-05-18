"""Integration tests for protocol adapters against mock agents."""

import asyncio

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from backend.adapters.http_adapter import HTTPAdapter
from backend.adapters.ws_adapter import WebSocketAdapter, create_adapter
from tests.e2e.mock_agent_http import app as mock_http_app, clear_sessions
from tests.e2e.mock_agent_ws import MockWebSocketServer


class MockHTTPAdapter(HTTPAdapter):
    """HTTP adapter that uses ASGI transport for testing."""

    async def start_session(self, scenario_id: str) -> str:
        session_id = self.generate_session_id()
        self._clients[session_id] = AsyncClient(
            transport=ASGITransport(app=mock_http_app),
            base_url="http://test",
            timeout=self.timeout,
        )
        return session_id


class TestHTTPAdapter:
    @pytest.fixture
    def http_adapter(self):
        return MockHTTPAdapter("http://test/chat", timeout=10.0)

    @pytest.fixture(autouse=True)
    def reset_mock_state(self):
        clear_sessions()
        yield
        clear_sessions()

    async def test_start_session(self, http_adapter):
        session_id = await http_adapter.start_session("test-scenario-1")
        assert session_id is not None
        assert len(session_id) > 0

    async def test_send_turn_basic(self, http_adapter):
        session_id = await http_adapter.start_session("test-scenario-2")
        turn = await http_adapter.send_turn(session_id, "Hello")

        assert turn.user_message == "Hello"
        assert turn.agent_response == "Hello! How can I help you today?"
        assert turn.error is None
        assert turn.latency_ms >= 0

    async def test_multiple_turns(self, http_adapter):
        session_id = await http_adapter.start_session("test-scenario-3")

        turn1 = await http_adapter.send_turn(session_id, "Hello")
        assert "Hello" in turn1.agent_response

        turn2 = await http_adapter.send_turn(session_id, "Tell me more")
        assert "understand" in turn2.agent_response.lower()

        turn3 = await http_adapter.send_turn(session_id, "Thanks")
        assert "found" in turn3.agent_response.lower()

    async def test_session_not_found(self, http_adapter):
        turn = await http_adapter.send_turn("nonexistent-session", "Hello")
        assert turn.error is not None
        assert "Session not found" in turn.error

    async def test_end_session(self, http_adapter):
        session_id = await http_adapter.start_session("test-scenario-4")
        await http_adapter.end_session(session_id)

        turn = await http_adapter.send_turn(session_id, "Hello")
        assert turn.error is not None


class TestWebSocketAdapter:
    @pytest.fixture
    async def ws_server(self):
        server = MockWebSocketServer(host="127.0.0.1", port=18766)
        await server.start()
        yield server
        await server.stop()

    @pytest.fixture
    def ws_adapter(self, ws_server):
        return WebSocketAdapter(ws_server.url, timeout=10.0)

    async def test_start_session(self, ws_adapter):
        session_id = await ws_adapter.start_session("test-scenario-ws-1")
        assert session_id is not None
        await ws_adapter.end_session(session_id)

    async def test_send_turn_basic(self, ws_adapter):
        session_id = await ws_adapter.start_session("test-scenario-ws-2")
        turn = await ws_adapter.send_turn(session_id, "Hello")

        assert turn.user_message == "Hello"
        assert turn.agent_response == "Hello! How can I help you today?"
        assert turn.error is None

        await ws_adapter.end_session(session_id)

    async def test_multiple_turns_ws(self, ws_adapter):
        session_id = await ws_adapter.start_session("test-scenario-ws-3")

        turn1 = await ws_adapter.send_turn(session_id, "Hello")
        assert "Hello" in turn1.agent_response

        turn2 = await ws_adapter.send_turn(session_id, "Tell me more")
        assert "understand" in turn2.agent_response.lower()

        await ws_adapter.end_session(session_id)

    async def test_booking_script_ws(self, ws_adapter):
        session_id = await ws_adapter.start_session("test-scenario-ws-4")

        turn1 = await ws_adapter.send_turn(session_id, "I want to make a booking")
        assert "booking" in turn1.agent_response.lower()

        turn2 = await ws_adapter.send_turn(session_id, "Tomorrow at 2pm")
        assert "availability" in turn2.agent_response.lower()

        await ws_adapter.end_session(session_id)

    async def test_session_not_found_ws(self, ws_adapter):
        turn = await ws_adapter.send_turn("nonexistent-session", "Hello")
        assert turn.error is not None
        assert "Session not found" in turn.error

    async def test_connection_failure(self):
        adapter = WebSocketAdapter("ws://127.0.0.1:19999", timeout=2.0)
        with pytest.raises(ConnectionError):
            await adapter.start_session("test-scenario")


class TestCreateAdapter:
    def test_create_http_adapter(self):
        adapter = create_adapter("http", "http://example.com/chat")
        assert isinstance(adapter, HTTPAdapter)
        assert adapter.url == "http://example.com/chat"

    def test_create_websocket_adapter(self):
        adapter = create_adapter("websocket", "ws://example.com/chat")
        assert isinstance(adapter, WebSocketAdapter)
        assert adapter.url == "ws://example.com/chat"

    def test_create_unknown_protocol_raises(self):
        with pytest.raises(ValueError, match="Unknown protocol"):
            create_adapter("grpc", "grpc://example.com")

    def test_create_with_custom_timeout(self):
        adapter = create_adapter("http", "http://example.com", timeout=60.0)
        assert adapter.timeout == 60.0


class TestTranscriptCreation:
    async def test_http_adapter_creates_transcript(self):
        adapter = HTTPAdapter("http://test/chat")
        transcript = adapter.create_transcript("scenario-1", "session-1", "http")

        assert transcript.scenario_id == "scenario-1"
        assert transcript.session_id == "session-1"
        assert transcript.protocol == "http"
        assert transcript.turns == []

    async def test_ws_adapter_creates_transcript(self):
        adapter = WebSocketAdapter("ws://test/chat")
        transcript = adapter.create_transcript("scenario-2", "session-2", "websocket")

        assert transcript.scenario_id == "scenario-2"
        assert transcript.session_id == "session-2"
        assert transcript.protocol == "websocket"
