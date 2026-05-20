import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import UTC, datetime

from backend.agents.world_state_patcher import (
    WorldStatePatcherAgent,
    world_state_patcher_node,
)
from backend.agents.state import ATAGraphState
from backend.llm.client import LLMResponse
from backend.models.suite import Scenario, ScenarioType, ScenarioVerdict, Verdict
from backend.models.transcript import Transcript, Turn
from backend.models.world_state import WorldState


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat = AsyncMock()
    return client


@pytest.fixture
def sample_world_state():
    return WorldState({
        "entities": [{"id": "user_1", "verified": True}],
        "catalog": {"slots": ["2026-05-20T10:00", "2026-05-20T14:00"]},
    })


@pytest.fixture
def sample_scenario():
    return Scenario(
        id="booking-scenario",
        type=ScenarioType.POSITIVE,
        description="Book a slot",
        turns=["Book 10:00 slot"],
    )


@pytest.fixture
def sample_transcript():
    return Transcript(
        scenario_id="booking-scenario",
        session_id="session-123",
        protocol="http",
        turns=[
            Turn(
                user_message="Book the 10:00 slot",
                agent_response="I've booked the 10:00 slot for you.",
                timestamp=datetime.now(UTC),
                latency_ms=100,
            ),
        ],
    )


@pytest.mark.asyncio
async def test_patcher_applies_patch_successfully(
    mock_llm_client, sample_world_state, sample_scenario, sample_transcript
):
    mock_llm_client.chat.return_value = LLMResponse(
        content="",
        tool_calls=[{
            "name": "patch_world_state",
            "arguments": {
                "ops": [
                    {"op": "remove", "path": "/catalog/slots/0"}
                ]
            },
            "id": "call-1",
        }],
    )

    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"booking-scenario": sample_transcript},
        verdicts={},
        world_state=sample_world_state,
        skipped_scenarios=set(),
        patch_ops={},
        world_state_snapshots={},
    )

    result = await world_state_patcher_node(state, mock_llm_client)

    assert result["status"] == "batch_patched"
    assert result["patch_failed"] is False
    assert "booking-scenario" in result["patch_ops"]
    assert len(result["world_state"].data["catalog"]["slots"]) == 1


@pytest.mark.asyncio
async def test_patcher_empty_ops_when_no_changes(
    mock_llm_client, sample_world_state, sample_scenario, sample_transcript
):
    mock_llm_client.chat.return_value = LLMResponse(
        content="",
        tool_calls=[{
            "name": "patch_world_state",
            "arguments": {"ops": []},
            "id": "call-1",
        }],
    )

    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"booking-scenario": sample_transcript},
        verdicts={},
        world_state=sample_world_state,
        skipped_scenarios=set(),
        patch_ops={},
        world_state_snapshots={},
    )

    result = await world_state_patcher_node(state, mock_llm_client)

    assert result["patch_failed"] is False
    assert result["patch_ops"]["booking-scenario"] == []


@pytest.mark.asyncio
async def test_patcher_validation_failure_cascades(mock_llm_client, sample_world_state):
    mock_llm_client.chat.return_value = LLMResponse(
        content="",
        tool_calls=[{
            "name": "patch_world_state",
            "arguments": {
                "ops": [
                    {"op": "remove", "path": "/nonexistent/path"}
                ]
            },
            "id": "call-1",
        }],
    )

    scenario1 = Scenario(
        id="scenario-1",
        type=ScenarioType.POSITIVE,
        description="First",
        turns=["Do something"],
    )
    scenario2 = Scenario(
        id="scenario-2",
        type=ScenarioType.POSITIVE,
        description="Second depends on first",
        turns=["Do another"],
        depends_on="scenario-1",
    )

    transcript1 = Transcript(
        scenario_id="scenario-1",
        session_id="s1",
        protocol="http",
        turns=[Turn(
            user_message="Do something",
            agent_response="Done",
            timestamp=datetime.now(UTC),
            latency_ms=50,
        )],
    )

    state = ATAGraphState(
        current_batch_scenarios=[scenario1],
        scenarios=[scenario1, scenario2],
        transcripts={"scenario-1": transcript1},
        verdicts={},
        world_state=sample_world_state,
        skipped_scenarios=set(),
        patch_ops={},
        world_state_snapshots={},
    )

    result = await world_state_patcher_node(state, mock_llm_client)

    assert result["patch_failed"] is True
    assert result["patch_failed_scenario_id"] == "scenario-1"
    assert "scenario-2" in result["skipped_scenarios"]
    assert result["verdicts"]["scenario-2"].verdict == Verdict.ERROR


@pytest.mark.asyncio
async def test_patcher_skips_errored_scenarios(
    mock_llm_client, sample_world_state, sample_scenario, sample_transcript
):
    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"booking-scenario": sample_transcript},
        verdicts={
            "booking-scenario": ScenarioVerdict(
                scenario_id="booking-scenario",
                verdict=Verdict.ERROR,
                reason="Connection failed",
            )
        },
        world_state=sample_world_state,
        skipped_scenarios=set(),
        patch_ops={},
        world_state_snapshots={},
    )

    result = await world_state_patcher_node(state, mock_llm_client)

    mock_llm_client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_patcher_skips_skipped_scenarios(
    mock_llm_client, sample_world_state, sample_scenario, sample_transcript
):
    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"booking-scenario": sample_transcript},
        verdicts={},
        world_state=sample_world_state,
        skipped_scenarios={"booking-scenario"},
        patch_ops={},
        world_state_snapshots={},
    )

    result = await world_state_patcher_node(state, mock_llm_client)

    mock_llm_client.chat.assert_not_called()


@pytest.mark.asyncio
async def test_patcher_creates_snapshots(
    mock_llm_client, sample_world_state, sample_scenario, sample_transcript
):
    mock_llm_client.chat.return_value = LLMResponse(
        content="",
        tool_calls=[{
            "name": "patch_world_state",
            "arguments": {"ops": []},
            "id": "call-1",
        }],
    )

    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"booking-scenario": sample_transcript},
        verdicts={},
        world_state=sample_world_state,
        skipped_scenarios=set(),
        patch_ops={},
        world_state_snapshots={},
    )

    result = await world_state_patcher_node(state, mock_llm_client)

    assert "booking-scenario_before" in result["world_state_snapshots"]
    assert "booking-scenario_after" in result["world_state_snapshots"]


@pytest.mark.asyncio
async def test_patcher_llm_error_cascades(mock_llm_client, sample_world_state, sample_scenario, sample_transcript):
    mock_llm_client.chat.side_effect = Exception("LLM API error")

    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"booking-scenario": sample_transcript},
        verdicts={},
        world_state=sample_world_state,
        skipped_scenarios=set(),
        patch_ops={},
        world_state_snapshots={},
    )

    result = await world_state_patcher_node(state, mock_llm_client)

    assert result["patch_failed"] is True
    assert "LLM error" in result["patch_failed_reason"]


@pytest.mark.asyncio
async def test_patcher_agent_class(
    mock_llm_client, sample_world_state, sample_scenario, sample_transcript
):
    mock_llm_client.chat.return_value = LLMResponse(
        content="",
        tool_calls=[{
            "name": "patch_world_state",
            "arguments": {"ops": []},
            "id": "call-1",
        }],
    )

    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"booking-scenario": sample_transcript},
        verdicts={},
        world_state=sample_world_state,
        skipped_scenarios=set(),
        patch_ops={},
        world_state_snapshots={},
    )

    agent = WorldStatePatcherAgent(mock_llm_client)
    result = await agent(state)

    assert result["status"] == "batch_patched"
