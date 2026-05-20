import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import UTC, datetime

from backend.agents.user_simulator import UserSimulatorAgent, user_simulator_node
from backend.agents.state import ATAGraphState
from backend.llm.client import LLMResponse
from backend.models.suite import Scenario, ScenarioType
from backend.models.transcript import Turn
from backend.models.world_state import WorldState


@pytest.fixture
def sample_world_state():
    return WorldState({
        "entities": [{"id": "user_1", "phone": "+1234567890"}],
        "catalog": {"slots": ["2026-05-20T10:00"]},
        "constraints": [],
        "context": {"language": "en"},
    })


@pytest.fixture
def sample_scenario():
    return Scenario(
        id="test-scenario",
        type=ScenarioType.POSITIVE,
        description="Test booking",
        turns=["I want to book a slot", "My phone is {{entities/0/phone}}"],
        persona="user_1",
    )


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.start_session = AsyncMock(return_value="session-123")
    adapter.send_turn = AsyncMock(
        return_value=Turn(
            user_message="test",
            agent_response="I can help with that",
            timestamp=datetime.now(UTC),
            latency_ms=100,
        )
    )
    adapter.end_session = AsyncMock()
    adapter.create_transcript = MagicMock()
    return adapter


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat = AsyncMock(
        return_value=LLMResponse(content="Adapted message")
    )
    return client


@pytest.mark.asyncio
async def test_user_simulator_basic_flow(
    sample_world_state, sample_scenario, mock_adapter, mock_llm_client
):
    from backend.models.transcript import Transcript

    mock_adapter.create_transcript.return_value = Transcript(
        scenario_id="test-scenario",
        session_id="session-123",
        protocol="http",
        turns=[],
    )

    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        world_state=sample_world_state,
        transcripts={},
        verdicts={},
        skipped_scenarios=set(),
    )

    result = await user_simulator_node(state, mock_llm_client, mock_adapter)

    assert result["status"] == "batch_executed"
    assert "test-scenario" in result["transcripts"]
    mock_adapter.start_session.assert_called_once()
    assert mock_adapter.send_turn.call_count == 2
    mock_adapter.end_session.assert_called_once()


@pytest.mark.asyncio
async def test_user_simulator_skips_skipped_scenarios(
    sample_world_state, sample_scenario, mock_adapter, mock_llm_client
):
    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        world_state=sample_world_state,
        transcripts={},
        verdicts={},
        skipped_scenarios={"test-scenario"},
    )

    result = await user_simulator_node(state, mock_llm_client, mock_adapter)

    assert result["status"] == "batch_executed"
    mock_adapter.start_session.assert_not_called()


@pytest.mark.asyncio
async def test_user_simulator_connection_error(
    sample_world_state, sample_scenario, mock_adapter, mock_llm_client
):
    from backend.models.suite import Verdict

    mock_adapter.start_session = AsyncMock(
        side_effect=ConnectionError("Failed to connect")
    )

    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        world_state=sample_world_state,
        transcripts={},
        verdicts={},
        skipped_scenarios=set(),
    )

    result = await user_simulator_node(state, mock_llm_client, mock_adapter)

    assert "test-scenario" in result["verdicts"]
    assert result["verdicts"]["test-scenario"].verdict == Verdict.ERROR


@pytest.mark.asyncio
async def test_user_simulator_turn_error(
    sample_world_state, sample_scenario, mock_adapter, mock_llm_client
):
    from backend.models.transcript import Transcript
    from backend.models.suite import Verdict

    mock_adapter.create_transcript.return_value = Transcript(
        scenario_id="test-scenario",
        session_id="session-123",
        protocol="http",
        turns=[],
    )
    mock_adapter.send_turn = AsyncMock(
        return_value=Turn(
            user_message="test",
            agent_response="",
            timestamp=datetime.now(UTC),
            latency_ms=100,
            error="TIMEOUT",
        )
    )

    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        world_state=sample_world_state,
        transcripts={},
        verdicts={},
        skipped_scenarios=set(),
    )

    result = await user_simulator_node(state, mock_llm_client, mock_adapter)

    assert "test-scenario" in result["verdicts"]
    assert result["verdicts"]["test-scenario"].verdict == Verdict.ERROR


@pytest.mark.asyncio
async def test_user_simulator_placeholder_resolution_error(
    sample_world_state, mock_adapter, mock_llm_client
):
    from backend.models.transcript import Transcript
    from backend.models.suite import Verdict

    bad_scenario = Scenario(
        id="bad-scenario",
        type=ScenarioType.POSITIVE,
        description="Test",
        turns=["{{invalid/path}}"],
    )

    mock_adapter.create_transcript.return_value = Transcript(
        scenario_id="bad-scenario",
        session_id="session-123",
        protocol="http",
        turns=[],
    )

    state = ATAGraphState(
        current_batch_scenarios=[bad_scenario],
        world_state=sample_world_state,
        transcripts={},
        verdicts={},
        skipped_scenarios=set(),
    )

    result = await user_simulator_node(state, mock_llm_client, mock_adapter)

    assert "bad-scenario" in result["verdicts"]
    assert result["verdicts"]["bad-scenario"].verdict == Verdict.ERROR
    assert "placeholder" in result["verdicts"]["bad-scenario"].reason.lower()


@pytest.mark.asyncio
async def test_user_simulator_multiple_scenarios_parallel(
    sample_world_state, mock_adapter, mock_llm_client
):
    from backend.models.transcript import Transcript

    scenarios = [
        Scenario(
            id=f"scenario-{i}",
            type=ScenarioType.POSITIVE,
            description="Test",
            turns=["Hello"],
        )
        for i in range(3)
    ]

    def create_transcript(scenario_id, session_id, protocol):
        return Transcript(
            scenario_id=scenario_id,
            session_id=session_id,
            protocol=protocol,
            turns=[],
        )

    mock_adapter.create_transcript.side_effect = create_transcript

    state = ATAGraphState(
        current_batch_scenarios=scenarios,
        world_state=sample_world_state,
        transcripts={},
        verdicts={},
        skipped_scenarios=set(),
    )

    result = await user_simulator_node(state, mock_llm_client, mock_adapter)

    assert len(result["transcripts"]) == 3


@pytest.mark.asyncio
async def test_user_simulator_agent_class(
    sample_world_state, sample_scenario, mock_adapter, mock_llm_client
):
    from backend.models.transcript import Transcript

    mock_adapter.create_transcript.return_value = Transcript(
        scenario_id="test-scenario",
        session_id="session-123",
        protocol="http",
        turns=[],
    )

    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        world_state=sample_world_state,
        transcripts={},
        verdicts={},
        skipped_scenarios=set(),
    )

    agent = UserSimulatorAgent(mock_llm_client, mock_adapter)
    result = await agent(state)

    assert result["status"] == "batch_executed"
