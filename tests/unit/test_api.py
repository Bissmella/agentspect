import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.dependencies import get_db
from backend.main import app


@pytest.fixture
def mock_session():
    return AsyncMock()


@pytest.fixture
async def client(mock_session):
    async def override_db():
        yield mock_session

    app.dependency_overrides[get_db] = override_db
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as c:
        yield c
    app.dependency_overrides.clear()


class TestCreateRun:
    async def test_invalid_yaml_returns_400(self, client):
        response = await client.post(
            "/api/runs",
            json={"yaml_content": "not:\n  valid:\n    yaml_missing_keys: true"},
        )
        assert response.status_code == 400

    async def test_missing_required_fields_returns_400(self, client):
        response = await client.post(
            "/api/runs",
            json={"yaml_content": "some_key: value"},
        )
        assert response.status_code == 400

    @patch("backend.main.run_suite_task")
    async def test_valid_yaml_creates_run(self, mock_task, client, mock_session):
        valid_yaml = """
agent_under_test:
  name: "Test Agent"
  url: "http://test/chat"
  protocol: "http"
  description: "A test agent"
  capabilities: ["chat"]
  known_limitations: []

world_state:
  entities: []
  catalog: {}
  constraints: []
  context: {}

test_config:
  total: 1
  positive: 1
  negative: 0

llm_config:
  provider: "openai"
  model: "gpt-4"
"""
        suite_id = uuid.uuid4()

        async def fake_refresh(obj):
            obj.id = suite_id

        mock_session.refresh = fake_refresh

        mock_task_result = MagicMock()
        mock_task_result.id = "celery-task-id"
        mock_task.delay.return_value = mock_task_result

        response = await client.post(
            "/api/runs",
            json={"yaml_content": valid_yaml},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["suite_id"] == str(suite_id)

        mock_session.add.assert_called_once()
        assert mock_session.commit.await_count >= 1


class TestGetRun:
    async def test_not_found_returns_404(self, client, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        response = await client.get(f"/api/runs/{uuid.uuid4()}")
        assert response.status_code == 404

    async def test_found_returns_suite(self, client, mock_session):
        suite_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        mock_suite = MagicMock()
        mock_suite.id = suite_id
        mock_suite.status.value = "running"
        mock_suite.created_at = now
        mock_suite.agent_name = "Test Agent"
        mock_suite.agent_url = "http://test"
        mock_suite.protocol.value = "http"
        mock_suite.llm_provider = "openai"
        mock_suite.llm_model = "gpt-4"
        mock_suite.total_scenarios = 5
        mock_suite.positive_count = 3
        mock_suite.negative_count = 2
        mock_suite.completed_at = None
        mock_suite.error_message = None
        mock_suite.scenarios = []

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = mock_suite
        mock_session.execute.return_value = mock_result

        response = await client.get(f"/api/runs/{suite_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["agent_name"] == "Test Agent"
        assert data["total_scenarios"] == 5
        assert data["status"] == "running"


class TestGetReport:
    async def test_suite_not_found_returns_404(self, client, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        response = await client.get(f"/api/runs/{uuid.uuid4()}/report")
        assert response.status_code == 404

    async def test_report_not_found_returns_404(self, client, mock_session):
        mock_suite = MagicMock()

        mock_result_suite = MagicMock()
        mock_result_suite.scalar_one_or_none.return_value = mock_suite

        mock_result_report = MagicMock()
        mock_result_report.scalar_one_or_none.return_value = None

        mock_session.execute.side_effect = [mock_result_suite, mock_result_report]

        response = await client.get(f"/api/runs/{uuid.uuid4()}/report")
        assert response.status_code == 404


class TestGetScenarios:
    async def test_suite_not_found_returns_404(self, client, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        response = await client.get(f"/api/runs/{uuid.uuid4()}/scenarios")
        assert response.status_code == 404


class TestGetTranscript:
    async def test_scenario_not_found_returns_404(self, client, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute.return_value = mock_result

        run_id = uuid.uuid4()
        sc_id = uuid.uuid4()
        response = await client.get(f"/api/runs/{run_id}/scenarios/{sc_id}/transcript")
        assert response.status_code == 404
