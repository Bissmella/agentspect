"""Capstone: a full voice_websocket suite through the real platform path.

Drives a tiny (total:1) voice suite against the in-repo voice mock agent and
asserts the platform preserves voice telemetry end to end:

    POST /api/runs  →  Celery task body (_execute_suite)  →  DB + S3  →  GET report/transcript

The engine's voice adapter is a real WebSocket *client*, so the mock runs on a
real loopback uvicorn server rather than an in-process ASGI shim (there is no
in-process WS client→ASGI bridge, and a real socket is the more faithful path).
Infra-gated: skips unless Postgres/Redis/MinIO are up and provider keys are set.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
import uvicorn
from httpx import ASGITransport, AsyncClient

from backend.config import settings
from backend.main import app as api_app
from backend.tasks import _execute_suite
from tests.e2e.voice_mock_agent import app as voice_app


def _tcp_ok(host: str | None, port: int | None, timeout: float = 0.5) -> bool:
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture
def requires_infra():
    import os

    missing: list[str] = []
    for env in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.getenv(env):
            missing.append(env)
    for label, url in (
        ("postgres", settings.database_url),
        ("redis", settings.redis_url),
        ("minio", settings.s3_endpoint_url),
    ):
        p = urlsplit(url)
        if not _tcp_ok(p.hostname, p.port):
            missing.append(label)
    if missing:
        pytest.skip(f"voice integration test needs: {', '.join(missing)}")


@pytest.fixture
def mock_server(tmp_path):
    """Serve the voice mock on a free loopback port for the run's lifetime."""
    db_path = tmp_path / "voice_db.json"
    db_path.write_text(json.dumps({"users": [], "reservations": []}))
    voice_app.state.db_path = str(db_path)
    voice_app.state.disability = None

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(voice_app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while not _tcp_ok("127.0.0.1", port) and time.time() < deadline:
        time.sleep(0.05)

    yield port, db_path

    server.should_exit = True
    thread.join(timeout=5)


def _voice_yaml(port: int) -> str:
    return f"""
agent_under_test:
  name: "Voice Reservation Agent"
  url: "ws://127.0.0.1:{port}/voice"
  protocol: "voice_websocket"
  description: >
    A voice reservation assistant that looks up users, registers new ones,
    and books available slots over a phone-style voice channel.
  capabilities:
    - "slot booking"
  known_limitations:
    - "does not support cancellations"

world_state:
  entities:
    - id: "user_alice"
      name: "Alice"
      phone: "+1111111111"
      registered: true
  catalog:
    slots:
      "2026-05-20T10:00": "available"
  constraints:
    - "only registered users can book a slot"
  context:
    language: "en"

test_config:
  total: 1
  positive: 1
  negative: 0

llm_config:
  provider: "openrouter"
  model: "openai/gpt-4o-mini"
"""


@pytest.mark.asyncio
async def test_voice_run_preserves_telemetry(requires_infra, mock_server):
    port, _ = mock_server
    yaml_str = _voice_yaml(port)

    transport = ASGITransport(app=api_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Real API creates the suite + runs the voice key guard; we run the task
        # body ourselves (apply_async patched so no live worker is required, and
        # to avoid asyncio.run() inside this already-running loop).
        with patch("backend.main.run_suite_task.apply_async",
                   return_value=SimpleNamespace(id="test-task")):
            resp = await client.post("/api/runs", json={"yaml_content": yaml_str})
        assert resp.status_code == 200, resp.text
        suite_id = resp.json()["suite_id"]

        # Celery task body against the real voice mock, real DB + S3.
        report = await _execute_suite(suite_id, yaml_str)
        assert report["verdict_counts"], report

        # Report surfaces voice metrics (always-registered; populated for voice).
        rep = await client.get(f"/api/runs/{suite_id}/report")
        assert rep.status_code == 200, rep.text
        metrics = rep.json()["summary"].get("metrics") or {}
        assert "time_to_first_audio" in metrics, metrics

        # At least one persisted turn carries a non-null voice block.
        scs = (await client.get(f"/api/runs/{suite_id}/scenarios")).json()
        assert scs, "no scenarios persisted"

        found_voice = False
        for sc in scs:
            tr = await client.get(f"/api/runs/{suite_id}/scenarios/{sc['id']}/transcript")
            if tr.status_code != 200:
                continue
            if any(t.get("voice") for t in tr.json()["turns"]):
                found_voice = True
                break
        assert found_voice, "no persisted turn had a voice telemetry block"
