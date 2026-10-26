import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import UTC, datetime

from agentspect.agents.scorer import (
    ScorerAgent,
    scorer_node,
    _check_probe_result,
    _evaluate_world_state_assertion_deterministic,
    _classify_recovery_quality,
    AssertionResult,
    RecoveryClassification,
)
from agentspect.models.suite import ScenarioVerdict
from agentspect.agents.state import ATAGraphState
from agentspect.models.suite import (
    BehavioralAssertion,
    DependsOnType,
    Scenario,
    ScenarioType,
    TranscriptAssertion,
    Verdict,
    WorldStateAssertion,
)
from agentspect.models.transcript import Transcript, Turn
from agentspect.models.world_state import WorldState


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


def _negative_refusal_scenario(sid="neg"):
    return Scenario(
        id=sid, type=ScenarioType.NEGATIVE, description="Unregistered caller tries to book",
        turns=["book me in"],
        assertions=[BehavioralAssertion(description="Agent refuses", expected_behavior="refusal")],
    )


def _one_turn_transcript(sid):
    return Transcript(
        scenario_id=sid, session_id="s", protocol="http",
        turns=[Turn(user_message="book me in", agent_response="That slot is taken, want 2pm?",
                    timestamp=datetime.now(UTC), latency_ms=10)],
    )


@pytest.mark.asyncio
async def test_negative_verbal_deflection_without_state_change_is_not_a_failure(mock_llm_client):
    # Agent didn't clearly refuse (offered an alternative) but never booked → not a breach.
    mock_llm_client.chat_with_structured_output.return_value = AssertionResult(
        satisfied=False, reasoning="Offered an alternative rather than refusing outright"
    )
    sc = _negative_refusal_scenario()
    state = ATAGraphState(
        current_batch_scenarios=[sc], scenarios=[sc],
        transcripts={"neg": _one_turn_transcript("neg")},
        verdicts={}, world_state_snapshots={}, skipped_scenarios=set(),
    )
    verdict = (await scorer_node(state, mock_llm_client))["verdicts"]["neg"]
    assert verdict.verdict == Verdict.SUCCESS
    assert "no forbidden action" in verdict.reason


@pytest.mark.asyncio
async def test_negative_not_refused_with_state_change_is_a_failure(mock_llm_client):
    # Same wording failure, but a state change proves the forbidden action happened → FAILURE.
    mock_llm_client.chat_with_structured_output.return_value = AssertionResult(
        satisfied=False, reasoning="Agent booked without registration"
    )
    sc = _negative_refusal_scenario("neg2")
    state = ATAGraphState(
        current_batch_scenarios=[sc], scenarios=[sc],
        transcripts={"neg2": _one_turn_transcript("neg2")},
        verdicts={},
        world_state_snapshots={"neg2_before": {"slots": ["A"]}, "neg2_after": {"slots": []}},
        skipped_scenarios=set(),
    )
    verdict = (await scorer_node(state, mock_llm_client))["verdicts"]["neg2"]
    assert verdict.verdict == Verdict.FAILURE


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


def _verdict(v):
    return ScenarioVerdict(scenario_id="x", verdict=v, reason="")


def _probe(dep_type):
    return Scenario(id="p", type=ScenarioType.POSITIVE, description="probe", turns=["x"],
                    depends_on="parent", depends_on_type=dep_type)


def test_probe_does_not_upgrade_a_parent_that_already_failed():
    # The headline bug: the agent never booked (parent FAILURE), the probe couldn't
    # rebook (FAILURE), and that must NOT be read as "booking persisted → SUCCESS".
    out = _check_probe_result(_probe(DependsOnType.PROBE), _verdict(Verdict.FAILURE), _verdict(Verdict.FAILURE))
    assert out == Verdict.FAILURE


def test_probe_confirms_or_suspects_only_an_unverified_parent():
    unv = _verdict(Verdict.SUCCESS_UNVERIFIED)
    # Probe's refusal assertion held (verdict SUCCESS) → agent refused the repeat →
    # state persisted → confirmed SUCCESS.
    assert _check_probe_result(_probe(DependsOnType.PROBE), _verdict(Verdict.SUCCESS), unv) == Verdict.SUCCESS
    # Probe re-did the action (refusal assertion failed → FAILURE) → state didn't hold → SUSPECT.
    assert _check_probe_result(_probe(DependsOnType.PROBE), _verdict(Verdict.FAILURE), unv) == Verdict.SUSPECT
    # An errored probe verifies nothing → parent stays unverified.
    assert _check_probe_result(_probe(DependsOnType.PROBE), _verdict(Verdict.ERROR), unv) == Verdict.SUCCESS_UNVERIFIED


def test_defensive_probe_only_applies_to_a_failed_negative():
    dp = _probe(DependsOnType.DEFENSIVE_PROBE)
    # Agent correctly refused (parent SUCCESS) → defensive probe can't downgrade it.
    assert _check_probe_result(dp, _verdict(Verdict.SUCCESS), _verdict(Verdict.SUCCESS)) == Verdict.SUCCESS
    # Agent wrongly accepted (parent FAILURE); clean-state assertion FAILED (corruption) → FAILURE_CORRUPT.
    assert _check_probe_result(dp, _verdict(Verdict.FAILURE), _verdict(Verdict.FAILURE)) == Verdict.FAILURE_CORRUPT
    # Clean state confirmed (assertion held → SUCCESS) → stays FAILURE, not corrupt.
    assert _check_probe_result(dp, _verdict(Verdict.SUCCESS), _verdict(Verdict.FAILURE)) == Verdict.FAILURE


@pytest.mark.asyncio
async def test_scorer_negative_success_gets_recovery_quality(mock_llm_client):
    mock_llm_client.chat_with_structured_output = AsyncMock(
        side_effect=[
            AssertionResult(satisfied=True, reasoning="Refused correctly"),
            RecoveryClassification(quality="clean_refusal", reasoning="Polite decline"),
        ]
    )

    scenario = Scenario(
        id="neg-1",
        type=ScenarioType.NEGATIVE,
        description="Should refuse",
        turns=["Bad request"],
        assertions=[
            BehavioralAssertion(description="Refuses", expected_behavior="refusal"),
        ],
    )
    transcript = Transcript(
        scenario_id="neg-1",
        session_id="s",
        protocol="http",
        turns=[Turn(
            user_message="Bad request",
            agent_response="Sorry, I can't do that.",
            timestamp=datetime.now(UTC),
            latency_ms=50,
        )],
    )
    state = ATAGraphState(
        current_batch_scenarios=[scenario],
        scenarios=[scenario],
        transcripts={"neg-1": transcript},
        verdicts={},
        world_state_snapshots={},
        skipped_scenarios=set(),
    )

    result = await scorer_node(state, mock_llm_client)
    verdict = result["verdicts"]["neg-1"]
    assert verdict.verdict == Verdict.SUCCESS
    assert verdict.recovery_quality == "clean_refusal"


@pytest.mark.asyncio
async def test_scorer_positive_has_no_recovery_quality(
    mock_llm_client, sample_scenario, sample_transcript
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
    verdict = result["verdicts"]["test-scenario"]
    assert verdict.recovery_quality is None


@pytest.mark.asyncio
async def test_scorer_recovery_quality_llm_failure(mock_llm_client):
    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AssertionResult(satisfied=True, reasoning="Refused")
        raise Exception("LLM error")

    mock_llm_client.chat_with_structured_output = AsyncMock(side_effect=side_effect)

    scenario = Scenario(
        id="neg-1",
        type=ScenarioType.NEGATIVE,
        description="Should refuse",
        turns=["Bad"],
        assertions=[
            BehavioralAssertion(description="Refuses", expected_behavior="refusal"),
        ],
    )
    transcript = Transcript(
        scenario_id="neg-1",
        session_id="s",
        protocol="http",
        turns=[Turn(
            user_message="Bad",
            agent_response="No",
            timestamp=datetime.now(UTC),
            latency_ms=50,
        )],
    )
    state = ATAGraphState(
        current_batch_scenarios=[scenario],
        scenarios=[scenario],
        transcripts={"neg-1": transcript},
        verdicts={},
        world_state_snapshots={},
        skipped_scenarios=set(),
    )

    result = await scorer_node(state, mock_llm_client)
    verdict = result["verdicts"]["neg-1"]
    assert verdict.verdict == Verdict.SUCCESS
    assert verdict.recovery_quality is None


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
