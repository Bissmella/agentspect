"""Toy voice agent that speaks the ATA voice-WS protocol.

A minimal, dependency-light target so the voice channel is runnable end to end.
It greets on connect, then parrots back whatever audio it receives — enough to
demonstrate the wire protocol and ATA's greeting capture + turn round-trip.

    python examples/voice_echo_agent.py    # serves ws://localhost:8765

Wire protocol (see ata/adapters/voice_ws_adapter.py):
    ATA  -> agent:  {"type": "audio", "data": "<base64>", "format": "wav"}
    agent-> ATA:    {"type": "audio", "data": "<base64>"}
    agent-> ATA:    {"type": "end_of_speech"}

Note: a *meaningful* transcript still needs STT/TTS keys on the ATA side (the
agent here just echoes audio bytes). This target's job is to exercise the
protocol, not to be a real booking bot.
"""

import asyncio
import json

import websockets

# A tiny silent-ish WAV header + no samples, base64-encoded, used as the greeting
# audio. Real targets would send synthesized speech here.
_GREETING_AUDIO_B64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA="


async def handler(connection):
    # Greet first, as phone agents do.
    await connection.send(json.dumps({"type": "audio", "data": _GREETING_AUDIO_B64}))
    await connection.send(json.dumps({"type": "end_of_speech"}))

    async for raw in connection:
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if msg.get("type") == "audio" and msg.get("data"):
            # Parrot the caller's audio back, then end the turn.
            await connection.send(json.dumps({"type": "audio", "data": msg["data"]}))
            await connection.send(json.dumps({"type": "end_of_speech"}))


async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Toy voice agent listening on ws://localhost:8765 (Ctrl+C to stop)")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
