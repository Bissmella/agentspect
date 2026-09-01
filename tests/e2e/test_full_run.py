"""
Phase 7 end-to-end test: the whole stack, front door to back.

POST YAML  →  Celery task body (orchestrator + real LLM against a mock agent)
           →  results stored in Postgres + S3
           →  GET status / report / scenarios / transcript
           →  assert verdicts, metrics, Postgres rows, and S3 blobs.

Unlike the unit API tests (which mock the DB session) this exercises the real
FastAPI app, the real async Postgres engine, the real S3/MinIO blob store, and
the real Redis pub/sub — exactly what `docker compose up` brings online. The
whole module therefore SKIPS unless that infrastructure is reachable and an
LLM key is configured, so the default `pytest` run stays green on a laptop with
nothing running.

To run it:
    docker compose up -d postgres redis minio
    export OPENROUTER_API_KEY=...        # or set it in .env
    uv run pytest tests/e2e/test_full_run.py -v
"""

import json
import socket
import uuid
from urllib.parse import urlparse

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch

from backend.config import settings


# ─────────────────────────────────────────────────────────────────
# Infra gate — decide at collection time whether the stack is up.
# ─────────────────────────────────────────────────────────────────
def _port_open(url: str, default_port: int) -> bool:
    parsed = urlparse(url if "//" in url else f"//{url}", scheme="tcp")
    host = parsed.hostname or "localhost"
    port = parsed.port or default_port
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


_DB_UP = _port_open(settings.database_url.split("@")[-1], 5432)
_REDIS_UP = _port_open(settings.redis_url, 6379)
_S3_UP = _port_open(settings.s3_endpoint_url, 9000)
_LLM_KEY = bool(settings.openrouter_api_key)

_INFRA_READY = _DB_UP and _REDIS_UP and _S3_UP and _LLM_KEY
_SKIP_REASON = (
    "e2e infra not available "
    f"(postgres={_DB_UP}, redis={_REDIS_UP}, minio={_S3_UP}, openrouter_key={_LLM_KEY}). "
    "Run `docker compose up -d postgres redis minio` and set OPENROUTER_API_KEY."
)

pytestmark = pytest.mark.skipif(not _INFRA_READY, reason=_SKIP_REASON)


# ─────────────────────────────────────────────────────────────────
# In-process HTTP adapter: routes ATA's calls to the mock agent app
# without needing a live server. This is a transport shim, not a mock
# of any ATA logic — the orchestrator, LLM, scorer, patcher all run for real.
# ─────────────────────────────────────────────────────────────────
def _in_process_adapter_factory(agent_app):
    from backend.adapters.http_adapter import HTTPAdapter

    class InProcessHTTPAdapter(HTTPAdapter):
        def __init__(self, url, app):
            super().__init__(url)
            self.app = app

        async def start_session(self, scenario_id: str) -> str:
            session_id = self.generate_session_id()
            self._clients[session_id] = AsyncClient(
                transport=ASGITransport(app=self.app),
                base_url="http://test",
            )
            return session_id

    return InProcessHTTPAdapter


TEST_YAML = """
agent_under_test:
  name: "E2E Reservation Agent"
  url: "http://test/chat"
  protocol: "http"
  description: >
    A reservation assistant that looks up users by phone, registers new users,
    and books available time slots. It refuses to double-book a taken slot.
  capabilities:
    - "user lookup by phone"
    - "new user registration"
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
      "2026-05-20T14:00": "available"
  constraints:
    - "only registered users can book a slot"
    - "a slot cannot be double-booked"
  context:
    language: "en"
    timezone: "Europe/Berlin"

test_config:
  total: 1
  positive: 1
  negative: 0

llm_config:
  provider: "openrouter"
  model: "openai/gpt-4o-mini"
"""


@pytest.fixture
async def db_tables():
    """Ensure the ORM tables exist on the real engine (idempotent)."""
    from backend.db.models import Base
    from backend.db.session import engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.fixture
def mock_agent(tmp_path):
    from tests.e2e.llm_mock_agent import app as agent_app
    from backend.llm.client import create_llm_client

    db_path = tmp_path / "e2e_db.json"
    db_path.write_text(json.dumps({
        "users": [{"name": "Alice", "phone": "+1111111111"}],
        "reservations": [],
    }))
    agent_app.state.db_path = str(db_path)
    agent_app.state.disability = None
    agent_app.state.llm_client = create_llm_client("openrouter", "openai/gpt-4o-mini")
    if hasattr(agent_app.state, "sessions"):
        agent_app.state.sessions.clear()

    yield agent_app, db_path

    if hasattr(agent_app.state, "sessions"):
        agent_app.state.sessions.clear()


@pytest.fixture
async def api_client():
    from backend.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


@pytest.mark.asyncio
async def test_full_run_end_to_end(db_tables, mock_agent, api_client):
    agent_app, db_path = mock_agent
    adapter_cls = _in_process_adapter_factory(agent_app)

    # 1. POST the YAML. Patch only the Celery hand-off (.delay needs a broker);
    #    we drive the task body ourselves below so the assertions are synchronous.
    with patch("backend.main.run_suite_task") as mock_task:
        fake_result = type("R", (), {"id": "e2e-task"})()
        mock_task.delay.return_value = fake_result

        resp = await api_client.post("/api/runs", json={"yaml_content": TEST_YAML})

    assert resp.status_code == 200, resp.text
    suite_id = resp.json()["suite_id"]
    assert resp.json()["status"] == "pending"

    # The suite row is really in Postgres, in PENDING state.
    from backend.db.models import Suite, SuiteStatus, Report
    from backend.db.session import async_session

    async with async_session() as session:
        suite = await session.get(Suite, uuid.UUID(suite_id))
        assert suite is not None
        assert suite.status == SuiteStatus.PENDING

    # 2. Run the actual task body: full orchestrator + real LLM + store to DB/S3.
    from backend import tasks

    with patch("backend.agents.orchestrator.create_adapter") as orch_adapter:
        orch_adapter.return_value = adapter_cls(url="http://test/chat", app=agent_app)
        await tasks._execute_suite(suite_id, TEST_YAML)

    # 3. GET status — suite completed with at least one scenario.
    resp = await api_client.get(f"/api/runs/{suite_id}")
    assert resp.status_code == 200
    status = resp.json()
    assert status["status"] == "completed"
    assert len(status["scenarios"]) >= 1

    # 4. GET report — quantitative metrics and verdict counts present.
    resp = await api_client.get(f"/api/runs/{suite_id}/report")
    assert resp.status_code == 200
    report = resp.json()
    summary = report["summary"]
    assert "verdict_counts" in summary
    assert "metrics" in summary and summary["metrics"] is not None
    assert "task_completion" in summary["metrics"]
    # The full report round-tripped through S3.
    assert report["full_report"] is not None
    assert "scenarios" in report["full_report"]

    # 5. GET scenarios — each has a verdict.
    resp = await api_client.get(f"/api/runs/{suite_id}/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert len(scenarios) >= 1
    assert all(s["verdict"] is not None for s in scenarios)

    # 6. GET a transcript from S3 for the first scenario that has one.
    for s in scenarios:
        r = await api_client.get(
            f"/api/runs/{suite_id}/scenarios/{s['id']}/transcript"
        )
        if r.status_code == 200:
            assert len(r.json()["turns"]) >= 1
            break

    # 7. Postgres has a Report row wired to the S3 blob key.
    async with async_session() as session:
        from sqlalchemy import select

        db_report = (
            await session.execute(select(Report).where(Report.suite_id == uuid.UUID(suite_id)))
        ).scalar_one_or_none()
        assert db_report is not None
        assert db_report.full_report_blob_key is not None

    # 8. The blob really exists in S3/MinIO.
    from backend.services.blob import async_download_json

    blob = await async_download_json(db_report.full_report_blob_key)
    assert blob["agent_name"] == "E2E Reservation Agent"
