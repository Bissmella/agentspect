"""
Mock WebSocket agent for testing the WebSocket protocol adapter.

A trivial WebSocket server that follows a scripted conversation.
Can be used as a test fixture or run standalone for manual testing.
"""

import asyncio
import json

import websockets


_scripts: dict[str, list[str]] = {
    "default": [
        "Hello! How can I help you today?",
        "I understand. Let me look into that for you.",
        "I've found the information you requested.",
        "Is there anything else I can help you with?",
        "Goodbye! Have a great day.",
    ],
    "booking": [
        "Welcome to the booking assistant. What date would you like to book?",
        "Let me check availability for that date.",
        "That slot is available. Would you like me to book it?",
        "Your booking has been confirmed. Confirmation number: BK12345.",
        "Thank you for using our service!",
    ],
    "refusal": [
        "I'm sorry, I cannot help with that request.",
        "That action is not permitted.",
        "I must decline. Is there something else I can help with?",
    ],
    "slow": [
        "Processing... please wait.",
    ],
}

_connection_count = 0


async def handle_connection(websocket):
    global _connection_count
    _connection_count += 1
    connection_id = _connection_count

    turn_index = 0
    script_name = "default"

    try:
        async for message in websocket:
            data = json.loads(message)
            user_message = data.get("message", "")

            if "booking" in user_message.lower():
                script_name = "booking"
            elif "refuse" in user_message.lower():
                script_name = "refusal"
            elif "slow" in user_message.lower():
                script_name = "slow"
                await asyncio.sleep(5)

            script = _scripts[script_name]

            if turn_index < len(script):
                agent_response = script[turn_index]
            else:
                agent_response = "I don't have anything more to say."

            turn_index += 1

            await websocket.send(
                json.dumps({"response": agent_response, "connection_id": connection_id})
            )

    except websockets.exceptions.ConnectionClosed:
        pass


class MockWebSocketServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8766):
        self.host = host
        self.port = port
        self._server = None
        self._task = None

    async def start(self):
        self._server = await websockets.serve(handle_connection, self.host, self.port)

    async def stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"


def set_script(name: str, responses: list[str]) -> None:
    _scripts[name] = responses


async def run_server(host: str = "127.0.0.1", port: int = 8766):
    server = MockWebSocketServer(host, port)
    await server.start()
    print(f"Mock WebSocket agent running at {server.url}")
    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        await server.stop()


if __name__ == "__main__":
    asyncio.run(run_server())
