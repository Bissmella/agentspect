import json
import time
from datetime import UTC, datetime

from websockets.exceptions import WebSocketException

from ata.adapters.base import ProtocolAdapter
from ata.adapters.ws_base import WebSocketConnectionMixin
from ata.models.transcript import Turn


class WebSocketAdapter(WebSocketConnectionMixin, ProtocolAdapter):
    async def start_session(self, scenario_id: str) -> str:
        session_id, _ = await self._connect_session("WebSocket")
        return session_id

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
                error=f"WebSocket error: {e!s}",
            )
        except json.JSONDecodeError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return Turn(
                user_message=message,
                agent_response="",
                timestamp=datetime.now(UTC),
                latency_ms=latency_ms,
                error=f"Invalid JSON response: {e!s}",
            )


def create_adapter(
    protocol: str,
    url: str,
    timeout: float = 30.0,
    voice=None,
) -> ProtocolAdapter:
    from ata.adapters.http_adapter import HTTPAdapter

    if protocol == "voice_websocket":
        return _create_voice_adapter(url=url, timeout=timeout, voice=voice)

    adapters = {
        "http": HTTPAdapter,
        "websocket": WebSocketAdapter,
    }

    if protocol not in adapters:
        raise ValueError(
            f"Unknown protocol: {protocol}. "
            f"Must be one of: {list(adapters.keys()) + ['voice_websocket']}"
        )

    return adapters[protocol](url=url, timeout=timeout)


def _create_voice_adapter(url: str, timeout: float, voice) -> ProtocolAdapter:
    from ata.adapters.voice_ws_adapter import VoiceWebSocketAdapter
    from ata.models.yaml_input import VoiceConfig
    from ata.voice.registry import create_voice_io

    voice = voice or VoiceConfig()
    voice_io = create_voice_io(
        stt_provider=voice.stt.provider,
        stt_model=voice.stt.model,
        tts_provider=voice.tts.provider,
        tts_model=voice.tts.model,
        voice=voice.tts.voice,
        language=voice.tts.language,
        accent=voice.tts.accent,
    )
    greeting_timeout = min(timeout, max(voice.endpointing.silence_ms / 1000.0, 1.0))
    return VoiceWebSocketAdapter(
        url=url,
        voice_io=voice_io,
        timeout=timeout,
        greeting_timeout=greeting_timeout,
    )
