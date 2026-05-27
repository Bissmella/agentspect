import json
from unittest.mock import patch
import pytest
from httpx import ASGITransport, AsyncClient

from backend.adapters.http_adapter import HTTPAdapter
from backend.agents.orchestrator import run_suite
from tests.e2e.llm_mock_agent import app as agent_app


# ─────────────────────────────────────────────────────────────────
# Route HTTP calls in-process to avoid needing a live server.
# This is a technical necessity, not a mock of ATA logic.
# ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────
# DB seed fixture — richer data
# ─────────────────────────────────────────────────────────────────
@pytest.fixture
def db_file(tmp_path):
    db_path = tmp_path / "reservation_db.json"
    db_path.write_text(json.dumps({
        "users": [
            {"name": "Alice", "phone": "+1111111111"},
            {"name": "Bob",   "phone": "+2222222222"},
        ],
        "reservations": []
    }, indent=2))
    agent_app.state.db_path = str(db_path)
    return db_path


@pytest.fixture(autouse=True)
def reset_sessions():
    if hasattr(agent_app.state, "sessions"):
        agent_app.state.sessions.clear()
    yield
    if hasattr(agent_app.state, "sessions"):
        agent_app.state.sessions.clear()


# ─────────────────────────────────────────────────────────────────
# Shared YAML — same world state for all test cases
# Rich set of entities and slots
# ─────────────────────────────────────────────────────────────────
TEST_YAML = """
agent_under_test:
  name: "LLM Reservation Agent"
  url: "http://test/chat"
  protocol: "http"
  description: >
    A reservation assistant that can look up users, register new users,
    and book available time slots. It refuses to book if the slot is already
    taken or the user cannot be registered.
  capabilities:
    - "user lookup by phone"
    - "new user registration"
    - "slot booking"
  known_limitations:
    - "does not support cancellations"
    - "does not support rescheduling"

world_state:
  entities:
    - id: "user_alice"
      name: "Alice"
      phone: "+1111111111"
      registered: true
    - id: "user_bob"
      name: "Bob"
      phone: "+2222222222"
      registered: true
    - id: "user_charlie"
      name: "Charlie"
      phone: "+3333333333"
      registered: false
  catalog:
    slots:
      "2026-06-01T09:00": "available"
      "2026-06-01T11:00": "available"
      "2026-06-01T14:00": "available"
      "2026-06-01T16:00": "available"
  constraints:
    - "only registered users can book a slot"
    - "a slot cannot be double-booked"
    - "cancellations and rescheduling are not supported"
  context:
    language: "en"
    timezone: "Europe/Berlin"

test_config:
  total: 4
  positive: 2
  negative: 2

llm_config:
  provider: "anthropic"
  model: "claude-sonnet-4-20250514"
"""


# ─────────────────────────────────────────────────────────────────
# Case 1: Healthy agent — no disability
# ATA generates scenarios, runs them, scores, patches, reports.
# All real LLM calls. We just assert on the report shape.
# ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@patch("backend.agents.orchestrator.create_adapter")
async def test_healthy_agent(mock_create_adapter, db_file):
    mock_create_adapter.return_value = InProcessHTTPAdapter(
        url="http://test/chat", app=agent_app
    )

    # Agent uses real LLM internally (create_llm_client is NOT patched)
    from backend.llm.client import create_llm_client
    agent_app.state.llm_client = create_llm_client("anthropic", "claude-sonnet-4-20250514")
    agent_app.state.disability = None

    report = await run_suite(TEST_YAML)

    # The report should exist and contain scenario results
    assert report is not None
    assert report["total_scenarios"] == 4
    assert "verdict_counts" in report
    assert "scenarios" in report
    assert len(report["scenarios"]) == 4

    # For a healthy agent, we expect mostly successes
    counts = report["verdict_counts"]
    failures = counts.get("failure", 0) + counts.get("error", 0)
    assert failures == 0, f"Healthy agent should not fail. Report:\n{json.dumps(report, indent=2)}"

    # Positive scenarios: slot booking should have mutated the DB
    db = json.loads(db_file.read_text())
    assert len(db["reservations"]) >= 1


# ─────────────────────────────────────────────────────────────────
# Case 2: Broken agent — user registration is disabled
# Charlie (unregistered) tries to book. Agent cannot register him.
# ATA should detect this as a failure in a positive scenario.
# ─────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
@patch("backend.agents.orchestrator.create_adapter")
async def test_agent_no_user_registration(mock_create_adapter, db_file):
    mock_create_adapter.return_value = InProcessHTTPAdapter(
        url="http://test/chat", app=agent_app
    )

    from backend.llm.client import create_llm_client
    agent_app.state.llm_client = create_llm_client("anthropic", "claude-sonnet-4-20250514")
    agent_app.state.disability = "no_user_registration"

    report = await run_suite(TEST_YAML)

    assert report is not None
    assert report["total_scenarios"] == 4

    # At least one scenario involving a new user should have failed
    counts = report["verdict_counts"]
    assert counts.get("failure", 0) >= 1, (
        "Registration disability should cause at least one scenario to fail.\n"
        f"Report:\n{json.dumps(report, indent=2)}"
    )

    # The failure analysis (generated by real reporter LLM) should exist
    assert report.get("failure_analysis") is not None

    # Charlie should NOT be in the database
    db = json.loads(db_file.read_text())
    assert not any(u["phone"] == "+3333333333" for u in db["users"])