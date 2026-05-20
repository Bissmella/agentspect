import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import UTC, datetime

from backend.agents.scorer import (
    ScorerAgent,
    scorer_node,
    _evaluate_world_state_assertion_deterministic,
    AssertionResult,
)
from backend.agents.state import ATAGraphState
from backend.models.suite import (
    BehavioralAssertion,
    DependsOnType,
    Scenario,
    ScenarioType,
    TranscriptAssertion,
    Verdict,
    WorldStateAssertion,
)
from backend.models.transcript import Transcript, Turn
from backend.models.world_state import WorldState


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat_with_structured_output = AsyncMock(
        return_value=AssertionResult(satisfied=True, reasoning="Check passed")
    )
    return client


class TestWorldStateAssertionDeterministic:
    def test_removed_operator_success(self):
        before = {"items": ["a", "b", "c"]}
        after = {"items": ["a", "b"]}
        assertion = WorldStateAssertion(
            description="Last item removed",
            path="/items/2",
            operator="removed",
        )

        result = _evaluate_world_state_assertion_deterministic(assertion, before, after)

        assert result.satisfied is True

    def test_removed_operator_failure(self):
        before = {"items": ["a", "b"]}
        after = {"items": ["a", "b"]}
        assertion = WorldStateAssertion(
            description="Item removed",
            path="/items/0",
            operator="removed",
        )

        result = _evaluate_world_state_assertion_deterministic(assertion, before, after)

        assert result.satisfied is False

    def test_added_operator_success(self):
        before = {"items": []}
        after = {"items": ["new"]}
        assertion = WorldStateAssertion(
            description="Item added",
            path="/items/0",
            operator="added",
        )

        result = _evaluate_world_state_assertion_deterministic(assertion, before, after)

        assert result.satisfied is True

    def test_equals_operator_success(self):
        before = {"status": "pending"}
        after = {"status": "completed"}
        assertion = WorldStateAssertion(
            description="Status is completed",
            path="/status",
            operator="equals",
            expected_value="completed",
        )

        result = _evaluate_world_state_assertion_deterministic(assertion, before, after)

        assert result.satisfied is True

    def test_equals_operator_failure(self):
        before = {}
        after = {"status": "pending"}
        assertion = WorldStateAssertion(
            description="Status is completed",
            path="/status",
            operator="equals",
            expected_value="completed",
        )

        result = _evaluate_world_state_assertion_deterministic(assertion, before, after)

        assert result.satisfied is False

    def test_contains_operator_success(self):
        before = {}
        after = {"tags": ["important", "urgent"]}
        assertion = WorldStateAssertion(
            description="Tags contain important",
            path="/tags",
            operator="contains",
            expected_value="important",
        )

        result = _evaluate_world_state_assertion_deterministic(assertion, before, after)

        assert result.satisfied is True

    def test_not_contains_operator_success(self):
        before = {}
        after = {"tags": ["normal"]}
        assertion = WorldStateAssertion(
            description="Tags do not contain urgent",
            path="/tags",
            operator="not_contains",
            expected_value="urgent",
        )

        result = _evaluate_world_state_assertion_deterministic(assertion, before, after)

        assert result.satisfied is True


@pytest.fixture
def sample_transcript():
    return Transcript(
        scenario_id="test-scenario",
        session_id="session-123",
        protocol="http",
        turns=[
            Turn(
                user_message="Book a slot for tomorrow",
                agent_response="I've booked slot A for you. Your confirmation number is ABC123.",
                timestamp=datetime.now(UTC),
                latency_ms=100,
            ),
        ],
    )


@pytest.fixture
def sample_scenario():
    return Scenario(
        id="test-scenario",
        type=ScenarioType.POSITIVE,
        description="Book a slot",
        turns=["Book a slot"],
        assertions=[
            BehavioralAssertion(
                description="Agent confirms booking",
                expected_behavior="confirmation",
            ),
        ],
    )


@pytest.mark.asyncio
async def test_scorer_positive_scenario_success(
    mock_llm_client, sample_transcript, sample_scenario
):
    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"test-scenario": sample_transcript},
        verdicts={},
        world_state_snapshots={
            "test-scenario_before": {"slots": ["A"]},
            "test-scenario_after": {"slots": []},
        },
        skipped_scenarios=set(),
    )

    result = await scorer_node(state, mock_llm_client)

    assert result["status"] == "batch_scored"
    assert "test-scenario" in result["verdicts"]
    verdict = result["verdicts"]["test-scenario"]
    assert verdict.verdict == Verdict.SUCCESS_UNVERIFIED


@pytest.mark.asyncio
async def test_scorer_negative_scenario_success(mock_llm_client, sample_transcript):
    mock_llm_client.chat_with_structured_output.return_value = AssertionResult(
        satisfied=True, reasoning="Agent refused as expected"
    )

    scenario = Scenario(
        id="negative-scenario",
        type=ScenarioType.NEGATIVE,
        description="Invalid request should be refused",
        turns=["Invalid request"],
        assertions=[
            BehavioralAssertion(
                description="Agent refuses",
                expected_behavior="refusal",
            ),
        ],
    )

    transcript = Transcript(
        scenario_id="negative-scenario",
        session_id="session-456",
        protocol="http",
        turns=[
            Turn(
                user_message="Invalid request",
                agent_response="I'm sorry, I cannot help with that.",
                timestamp=datetime.now(UTC),
                latency_ms=50,
            ),
        ],
    )

    state = ATAGraphState(
        current_batch_scenarios=[scenario],
        scenarios=[scenario],
        transcripts={"negative-scenario": transcript},
        verdicts={},
        world_state_snapshots={},
        skipped_scenarios=set(),
    )

    result = await scorer_node(state, mock_llm_client)

    verdict = result["verdicts"]["negative-scenario"]
    assert verdict.verdict == Verdict.SUCCESS


@pytest.mark.asyncio
async def test_scorer_no_transcript(mock_llm_client, sample_scenario):
    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={},
        verdicts={},
        world_state_snapshots={},
        skipped_scenarios=set(),
    )

    result = await scorer_node(state, mock_llm_client)

    verdict = result["verdicts"]["test-scenario"]
    assert verdict.verdict == Verdict.ERROR
    assert "No transcript" in verdict.reason


@pytest.mark.asyncio
async def test_scorer_skipped_scenario(mock_llm_client, sample_scenario, sample_transcript):
    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"test-scenario": sample_transcript},
        verdicts={},
        world_state_snapshots={},
        skipped_scenarios={"test-scenario"},
    )

    result = await scorer_node(state, mock_llm_client)

    assert "test-scenario" not in result["verdicts"]


@pytest.mark.asyncio
async def test_scorer_with_probe_chain(mock_llm_client):
    primary = Scenario(
        id="primary",
        type=ScenarioType.POSITIVE,
        description="Book slot",
        turns=["Book"],
        assertions=[
            WorldStateAssertion(
                description="Slot removed",
                path="/slots/0",
                operator="removed",
            )
        ],
    )

    probe = Scenario(
        id="probe",
        type=ScenarioType.POSITIVE,
        description="Try to book again",
        turns=["Book same slot"],
        depends_on="primary",
        depends_on_type=DependsOnType.PROBE,
        assertions=[
            BehavioralAssertion(
                description="Agent refuses",
                expected_behavior="refusal",
            )
        ],
    )

    mock_llm_client.chat_with_structured_output.return_value = AssertionResult(
        satisfied=True, reasoning="Passed"
    )

    state = ATAGraphState(
        current_batch_scenarios=[primary, probe],
        scenarios=[primary, probe],
        transcripts={
            "primary": Transcript(
                scenario_id="primary",
                session_id="s1",
                protocol="http",
                turns=[Turn(
                    user_message="Book",
                    agent_response="Booked!",
                    timestamp=datetime.now(UTC),
                    latency_ms=50,
                )],
            ),
            "probe": Transcript(
                scenario_id="probe",
                session_id="s2",
                protocol="http",
                turns=[Turn(
                    user_message="Book same slot",
                    agent_response="Sorry, that slot is taken",
                    timestamp=datetime.now(UTC),
                    latency_ms=50,
                )],
            ),
        },
        verdicts={},
        world_state_snapshots={
            "primary_before": {"slots": ["A"]},
            "primary_after": {"slots": []},
        },
        skipped_scenarios=set(),
    )

    result = await scorer_node(state, mock_llm_client)

    assert "primary" in result["verdicts"]
    assert "probe" in result["verdicts"]


@pytest.mark.asyncio
async def test_scorer_agent_class(mock_llm_client, sample_scenario, sample_transcript):
    state = ATAGraphState(
        current_batch_scenarios=[sample_scenario],
        scenarios=[sample_scenario],
        transcripts={"test-scenario": sample_transcript},
        verdicts={},
        world_state_snapshots={},
        skipped_scenarios=set(),
    )

    agent = ScorerAgent(mock_llm_client)
    result = await agent(state)

    assert result["status"] == "batch_scored"
