import uuid
from datetime import datetime, timezone

from backend.schemas.run import (
    CreateRunRequest,
    CreateRunResponse,
    ReportResponse,
    ScenarioDetailResponse,
    ScenarioSummary,
    SuiteStatusResponse,
    TranscriptResponse,
    TurnResponse,
)


class TestCreateRunModels:
    def test_request(self):
        req = CreateRunRequest(yaml_content="agent_under_test:\n  name: test")
        assert req.yaml_content == "agent_under_test:\n  name: test"

    def test_response(self):
        sid = uuid.uuid4()
        resp = CreateRunResponse(suite_id=sid)
        assert resp.suite_id == sid
        assert resp.status == "pending"


class TestSuiteStatusResponse:
    def test_full_response(self):
        sid = uuid.uuid4()
        now = datetime.now(timezone.utc)
        resp = SuiteStatusResponse(
            id=sid,
            status="running",
            created_at=now,
            agent_name="Test Agent",
            agent_url="http://test",
            protocol="http",
            llm_provider="openai",
            llm_model="gpt-4",
            total_scenarios=10,
            positive_count=7,
            negative_count=3,
            scenarios=[],
        )
        assert resp.id == sid
        assert resp.total_scenarios == 10
        assert resp.error_message is None

    def test_with_scenarios(self):
        sc_id = uuid.uuid4()
        sc = ScenarioSummary(
            id=sc_id,
            type="positive",
            description="test scenario",
            status="completed",
            verdict="success",
        )
        assert sc.verdict == "success"
        assert sc.verdict_reason is None


class TestScenarioDetailResponse:
    def test_minimal(self):
        sc = ScenarioDetailResponse(
            id=uuid.uuid4(),
            type="negative",
            description="test",
            status="completed",
        )
        assert sc.verdict is None
        assert sc.depends_on is None


class TestTranscriptResponse:
    def test_with_turns(self):
        t = TurnResponse(user="hello", agent="hi there", latency_ms=150)
        resp = TranscriptResponse(scenario_id=uuid.uuid4(), turns=[t])
        assert len(resp.turns) == 1
        assert resp.turns[0].latency_ms == 150


class TestReportResponse:
    def test_with_summary(self):
        resp = ReportResponse(
            suite_id=uuid.uuid4(),
            created_at=datetime.now(timezone.utc),
            summary={"verdict_counts": {"success": 5}},
        )
        assert resp.summary["verdict_counts"]["success"] == 5
        assert resp.full_report is None
