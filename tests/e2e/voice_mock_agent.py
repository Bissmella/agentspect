"""Voice agent-under-test mock speaking the voice-WS frame protocol.

The engine's voice adapter connects here as a client and, per turn, streams the
simulated user's speech as ``{"type":"audio","data":<b64 wav>,"format":"wav"}``
frames followed by ``{"type":"end_of_speech"}``. This mock mirrors that back:
it STTs the inbound audio, runs the *same* agent brain as the HTTP mock
(``run_turn``), TTSs the reply, and returns one audio frame + ``end_of_speech``.

One WS connection == one scenario == one session (history lives for the socket).
Provider keys come from the environment via agentspect's OpenAI voice providers.
"""

from __future__ import annotations

import base64
import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agentspect import create_llm_client
from agentspect.voice import create_voice_io  # importing registers OpenAI providers

from tests.e2e.llm_mock_agent import run_turn

app = FastAPI(title="Voice Mock Agent Under Test")
logger = logging.getLogger(__name__)

_END_SIGNALS = {"end_of_speech", "end", "eos"}


def _llm(ws: WebSocket):
    return getattr(ws.app.state, "llm_client", None) or create_llm_client(
        "openrouter", "openai/gpt-4o-mini"
    )


def _voice_io(ws: WebSocket):
    return getattr(ws.app.state, "voice_io", None) or create_voice_io(
        stt_provider="openai", tts_provider="openai"
    )


async def _recv_utterance(ws: WebSocket) -> bytes:
    """Accumulate inbound audio frames until end_of_speech; raise on disconnect."""
    audio = bytearray()
    while True:
        msg = json.loads(await ws.receive_text())
        if msg.get("type") in _END_SIGNALS:
            return bytes(audio)
        if msg.get("type") == "audio" and msg.get("data"):
            audio.extend(base64.b64decode(msg["data"]))


async def _send_utterance(ws: WebSocket, audio: bytes) -> None:
    await ws.send_text(json.dumps(
        {"type": "audio", "data": base64.b64encode(audio).decode("ascii"), "format": "wav"}
    ))
    await ws.send_text(json.dumps({"type": "end_of_speech"}))


@app.websocket("/voice")
async def voice(ws: WebSocket):
    await ws.accept()
    db_path = getattr(ws.app.state, "db_path", "llm_agent_db.json")
    disability = getattr(ws.app.state, "disability", None)
    llm_client = _llm(ws)
    voice_io = _voice_io(ws)
    history: list[dict] = []

    try:
        while True:
            audio_in = await _recv_utterance(ws)
            if not audio_in:
                continue  # empty turn — wait for the next (or a disconnect)
            try:
                user_text = (await voice_io.transcribe(audio_in)).text
                reply = await run_turn(llm_client, history, user_text, db_path, disability)
                await _send_utterance(ws, await voice_io.synthesize(reply))
            except Exception:
                # Surface the real STT/TTS/LLM cause instead of a silent drop —
                # otherwise the engine only sees a disconnect and tests report
                # the misleading "no voice block".
                logger.exception("voice mock turn failed (%d bytes in)", len(audio_in))
                raise
    except WebSocketDisconnect:
        pass  # scenario done — ATA closed the socket
