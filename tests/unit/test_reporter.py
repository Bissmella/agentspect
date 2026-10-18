import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import UTC, datetime

from agentspect.agents.reporter import ReporterAgent, reporter_node, FailureAnalysis
from agentspect.agents.state import ATAGraphState
from agentspect.models.suite import (
    DependsOnType,
    Scenario,
    ScenarioType,
    ScenarioVerdict,
    Verdict,
)
from agentspect.models.transcript import (
    Transcript,
    Turn,
    VoiceEvent,
    VoiceEventKind,
    VoiceMeta,
)
from agentspect.models.yaml_input import AgentUnderTest, WorldStateInput


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat_with_structured_output = AsyncMock(
        return_value=FailureAnalysis(
            summary="Some tests failed",
            constraint_violations=["Constraint X violated"],
            entity_lookup_failures=[],
            catalog_misuse=[],
            recommendations=["Fix constraint X handling"],
        )
    )
    return client


@pytest.fixture
def sample_scenarios():
    return [
        Scenario(
            id="positive-1",
            type=ScenarioType.POSITIVE,
            description="Book a slot",
            turns=["Book slot A"],
        ),
        Scenario(
            id="positive-2",
            type=ScenarioType.POSITIVE,
            description="Verify booking",
            turns=["Check booking"],
            depends_on="positive-1",
            depends_on_type=DependsOnType.PROBE,
        ),
        Scenario(
            id="negative-1",
            type=ScenarioType.NEGATIVE,
            description="Invalid request",
            turns=["Invalid"],
        ),
    ]


@pytest.fixture
def sample_transcripts():
    return {
        "positive-1": Transcript(
            scenario_id="positive-1",
            session_id="s1",
            protocol="http",
            turns=[
                Turn(
                    user_message="Book slot A",
                    agent_response="Booked!",
                    timestamp=datetime.now(UTC),
                    latency_ms=100,
                )
            ],
        ),
        "positive-2": Transcript(
            scenario_id="positive-2",
            session_id="s2",
            protocol="http",
            turns=[
                Turn(
                    user_message="Check booking",
                    agent_response="You have a booking for slot A",
                    timestamp=datetime.now(UTC),
                    latency_ms=80,
                )
            ],
        ),
        "negative-1": Transcript(
            scenario_id="negative-1",
            session_id="s3",
            protocol="http",
            turns=[
                Turn(
                    user_message="Invalid",
                    agent_response="I cannot help with that",
                    timestamp=datetime.now(UTC),
                    latency_ms=50,
                )
            ],
        ),
    }


@pytest.fixture
def sample_verdicts():
    return {
        "positive-1": ScenarioVerdict(
            scenario_id="positive-1",
            verdict=Verdict.SUCCESS,
            reason="All assertions passed",
            assertion_results=[{"satisfied": True}],
        ),
        "positive-2": ScenarioVerdict(
            scenario_id="positive-2",
            verdict=Verdict.SUCCESS,
            reason="Probe confirmed",
            assertion_results=[],
        ),
        "negative-1": ScenarioVerdict(
            scenario_id="negative-1",
            verdict=Verdict.SUCCESS,
            reason="Agent refused as expected",
            assertion_results=[{"satisfied": True}],
        ),
    }


@pytest.mark.asyncio
async def test_reporter_generates_complete_report(
    mock_llm_client, sample_scenarios, sample_transcripts, sample_verdicts
):
    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test Agent",
            url="http://localhost:8000",
            protocol="http",
            description="Test",
        ),
        world_state_input=WorldStateInput(),
        scenarios=sample_scenarios,
        transcripts=sample_transcripts,
        verdicts=sample_verdicts,
        world_state_snapshots={
            "positive-1_before": {"slots": ["A", "B"]},
            "positive-1_after": {"slots": ["B"]},
        },
        patch_ops={"positive-1": [{"op": "remove", "path": "/slots/0"}]},
    )

    result = await reporter_node(state, mock_llm_client)

    assert result["status"] == "completed"
    report = result["report"]

    assert report["agent_name"] == "Test Agent"
    assert report["total_scenarios"] == 3
    assert "verdict_counts" in report
    assert report["verdict_counts"]["success"] == 3


@pytest.mark.asyncio
async def test_reporter_includes_probe_chains(
    mock_llm_client, sample_scenarios, sample_transcripts, sample_verdicts
):
    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test Agent",
            url="http://localhost:8000",
            protocol="http",
            description="Test",
        ),
        scenarios=sample_scenarios,
        transcripts=sample_transcripts,
        verdicts=sample_verdicts,
        world_state_snapshots={},
        patch_ops={},
    )

    result = await reporter_node(state, mock_llm_client)

    report = result["report"]
    assert len(report["probe_chains"]) == 1
    assert report["probe_chains"][0]["parent_id"] == "positive-1"
    assert report["probe_chains"][0]["probe_id"] == "positive-2"


@pytest.mark.asyncio
async def test_reporter_includes_world_state_audit(
    mock_llm_client, sample_scenarios, sample_transcripts, sample_verdicts
):
    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test Agent",
            url="http://localhost:8000",
            protocol="http",
            description="Test",
        ),
        scenarios=sample_scenarios,
        transcripts=sample_transcripts,
        verdicts=sample_verdicts,
        world_state_snapshots={
            "positive-1_before": {"slots": ["A"]},
            "positive-1_after": {"slots": []},
        },
        patch_ops={"positive-1": [{"op": "remove", "path": "/slots/0"}]},
    )

    result = await reporter_node(state, mock_llm_client)

    report = result["report"]
    assert len(report["world_state_audit"]) >= 1
    audit_entry = report["world_state_audit"][0]
    assert audit_entry["scenario_id"] == "positive-1"
    assert audit_entry["before"] == {"slots": ["A"]}
    assert audit_entry["after"] == {"slots": []}


@pytest.mark.asyncio
async def test_reporter_generates_failure_analysis(mock_llm_client, sample_scenarios):
    failed_verdicts = {
        "positive-1": ScenarioVerdict(
            scenario_id="positive-1",
            verdict=Verdict.FAILURE,
            reason="Assertion failed",
        ),
        "negative-1": ScenarioVerdict(
            scenario_id="negative-1",
            verdict=Verdict.SUCCESS,
            reason="OK",
        ),
    }

    failed_transcript = Transcript(
        scenario_id="positive-1",
        session_id="s1",
        protocol="http",
        turns=[
            Turn(
                user_message="Book",
                agent_response="Error occurred",
                timestamp=datetime.now(UTC),
                latency_ms=100,
            )
        ],
    )

    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test Agent",
            url="http://localhost:8000",
            protocol="http",
            description="Test",
        ),
        world_state_input=WorldStateInput(
            constraints=["Only verified users can book"]
        ),
        scenarios=sample_scenarios[:2],
        transcripts={"positive-1": failed_transcript, "negative-1": Transcript(
            scenario_id="negative-1",
            session_id="s2",
            protocol="http",
            turns=[],
        )},
        verdicts=failed_verdicts,
        world_state_snapshots={},
        patch_ops={},
    )

    result = await reporter_node(state, mock_llm_client)

    report = result["report"]
    assert report["failure_analysis"] is not None
    assert "summary" in report["failure_analysis"]


@pytest.mark.asyncio
async def test_reporter_no_failure_analysis_when_all_pass(
    mock_llm_client, sample_scenarios, sample_transcripts, sample_verdicts
):
    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test Agent",
            url="http://localhost:8000",
            protocol="http",
            description="Test",
        ),
        world_state_input=WorldStateInput(),
        scenarios=sample_scenarios,
        transcripts=sample_transcripts,
        verdicts=sample_verdicts,
        world_state_snapshots={},
        patch_ops={},
    )

    result = await reporter_node(state, mock_llm_client)

    report = result["report"]
    assert report["failure_analysis"] is None


@pytest.mark.asyncio
async def test_reporter_handles_missing_agent_info(
    mock_llm_client, sample_scenarios, sample_transcripts, sample_verdicts
):
    state = ATAGraphState(
        scenarios=sample_scenarios,
        transcripts=sample_transcripts,
        verdicts=sample_verdicts,
        world_state_snapshots={},
        patch_ops={},
    )

    result = await reporter_node(state, mock_llm_client)

    report = result["report"]
    assert report["agent_name"] == "Unknown"
    assert report["agent_url"] is None


@pytest.mark.asyncio
async def test_reporter_includes_metrics(
    mock_llm_client, sample_scenarios, sample_transcripts, sample_verdicts
):
    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test Agent",
            url="http://localhost:8000",
            protocol="http",
            description="Test",
        ),
        world_state_input=WorldStateInput(),
        scenarios=sample_scenarios,
        transcripts=sample_transcripts,
        verdicts=sample_verdicts,
        world_state_snapshots={},
        patch_ops={},
    )

    result = await reporter_node(state, mock_llm_client)

    report = result["report"]
    assert "metrics" in report
    metrics = report["metrics"]
    assert "task_completion" in metrics
    assert "boundary_adherence" in metrics
    assert "verification_rate" in metrics
    assert "constraint_violations" in metrics
    assert "recovery_behavior" in metrics
    assert "conversation_efficiency" in metrics
    assert metrics["task_completion"]["total"] == 1
    assert metrics["boundary_adherence"]["total"] == 1


@pytest.mark.asyncio
async def test_reporter_serializes_per_turn_voice(mock_llm_client, sample_scenarios):
    """Voice telemetry must survive into the serialized turns (the UI reads it)."""
    voice_transcript = Transcript(
        scenario_id="positive-1",
        session_id="s1",
        protocol="voice_websocket",
        turns=[
            Turn(
                user_message="Book slot A",
                agent_response="Booked!",
                latency_ms=100,
                voice=VoiceMeta(time_to_first_audio_ms=420, agent_speech_ms=1300, stt_confidence=0.91),
                voice_events=[VoiceEvent(t_ms=0, kind=VoiceEventKind.AGENT_SPEECH_START, party="agent")],
            )
        ],
    )
    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Voice Agent", url="tel:+15551234567", protocol="voice_websocket", description="phone"
        ),
        scenarios=sample_scenarios[:1],
        transcripts={"positive-1": voice_transcript},
        verdicts={"positive-1": ScenarioVerdict(
            scenario_id="positive-1", verdict=Verdict.SUCCESS, reason="ok", assertion_results=[]
        )},
        world_state_snapshots={},
        patch_ops={},
    )

    turn = (await reporter_node(state, mock_llm_client))["report"]["scenarios"][0]["turns"][0]
    assert turn["voice"]["time_to_first_audio_ms"] == 420
    assert turn["voice_events"][0]["kind"] == VoiceEventKind.AGENT_SPEECH_START


@pytest.mark.asyncio
async def test_reporter_text_turn_has_null_voice(
    mock_llm_client, sample_scenarios, sample_transcripts, sample_verdicts
):
    """Text runs carry voice=None, not a crash or a fabricated block."""
    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test Agent", url="http://localhost:8000", protocol="http", description="Test"
        ),
        scenarios=sample_scenarios,
        transcripts=sample_transcripts,
        verdicts=sample_verdicts,
        world_state_snapshots={},
        patch_ops={},
    )
    turn = (await reporter_node(state, mock_llm_client))["report"]["scenarios"][0]["turns"][0]
    assert turn["voice"] is None
    assert turn["voice_events"] == []


@pytest.mark.asyncio
async def test_reporter_agent_class(
    mock_llm_client, sample_scenarios, sample_transcripts, sample_verdicts
):
    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test Agent",
            url="http://localhost:8000",
            protocol="http",
            description="Test",
        ),
        scenarios=sample_scenarios,
        transcripts=sample_transcripts,
        verdicts=sample_verdicts,
        world_state_snapshots={},
        patch_ops={},
    )

    agent = ReporterAgent(mock_llm_client)
    result = await agent(state)

    assert result["status"] == "completed"
    assert "report" in result
