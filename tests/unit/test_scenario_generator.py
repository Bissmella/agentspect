import pytest
from unittest.mock import AsyncMock, MagicMock

from agentspect.agents.scenario_generator import (
    ScenarioGeneratorAgent,
    scenario_generator_node,
    GeneratedScenarios,
    GeneratedScenario,
    GeneratedAssertion,
)
from agentspect.agents.state import ATAGraphState
from agentspect.models.suite import ScenarioType
from agentspect.models.yaml_input import AgentUnderTest, LLMConfig, TestConfig, WorldStateInput


@pytest.fixture
def sample_state() -> ATAGraphState:
    return ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test Agent",
            url="http://localhost:8000",
            protocol="http",
            description="A test booking agent",
            capabilities=["booking", "lookup"],
            known_limitations=["no rescheduling"],
        ),
        world_state_input=WorldStateInput(
            entities=[
                {"id": "user_1", "phone": "+1234567890", "name": "John"},
            ],
            catalog={"slots": ["2026-05-20T10:00", "2026-05-20T14:00"]},
            constraints=["only verified users can book"],
            context={"language": "en"},
        ),
        test_config=TestConfig(total=2, positive=1, negative=1),
        llm_config=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
    )


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.chat_with_structured_output = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_scenario_generator_success(sample_state, mock_llm_client):
    mock_llm_client.chat_with_structured_output.return_value = GeneratedScenarios(
        scenarios=[
            GeneratedScenario(
                id="positive-booking-1",
                type="positive",
                description="User books available slot",
                turns=["I want to book a slot", "{{entities/0/phone}}"],
                persona="user_1",
                assertions=[
                    GeneratedAssertion(
                        type="behavioral",
                        description="Agent confirms booking",
                        expected_behavior="confirmation",
                    )
                ],
            ),
            GeneratedScenario(
                id="negative-unknown-user",
                type="negative",
                description="Unknown user attempts booking",
                turns=["I want to book", "+9999999999"],
                assertions=[
                    GeneratedAssertion(
                        type="behavioral",
                        description="Agent refuses",
                        expected_behavior="refusal",
                    )
                ],
            ),
        ]
    )

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "scenarios_generated"
    assert len(result["scenarios"]) == 2
    assert result["scenarios"][0].type == ScenarioType.POSITIVE
    assert result["scenarios"][1].type == ScenarioType.NEGATIVE
    assert "scenario_batches" in result


@pytest.mark.asyncio
async def test_scenario_generator_short_count_warns_not_fails(sample_state, mock_llm_client):
    # LLM keeps returning only 1 positive (0 negative) though 1/1 was asked. After the
    # retries it accepts what it has with a warning — it must NOT nuke the whole suite.
    mock_llm_client.chat_with_structured_output.return_value = GeneratedScenarios(
        scenarios=[
            GeneratedScenario(id="positive-1", type="positive", description="Test", turns=["Hello"]),
        ]
    )

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "scenarios_generated"
    assert len(result["scenarios"]) == 1
    assert "0 of 1 negative" in result.get("warning", "")
    assert mock_llm_client.chat_with_structured_output.await_count >= 2  # retried first


@pytest.mark.asyncio
async def test_scenario_generator_tops_up_on_retry(sample_state, mock_llm_client):
    # First call is short (positive only); the top-up retry supplies the missing negative,
    # so the suite reaches the full count with no warning.
    first = GeneratedScenarios(scenarios=[
        GeneratedScenario(id="p1", type="positive", description="book", turns=["Hi"]),
    ])
    second = GeneratedScenarios(scenarios=[
        GeneratedScenario(id="n1", type="negative", description="refuse", turns=["bad"]),
    ])
    mock_llm_client.chat_with_structured_output = AsyncMock(side_effect=[first, second])

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "scenarios_generated"
    assert result.get("warning") is None
    assert {s.id for s in result["scenarios"]} == {"p1", "n1"}
    assert mock_llm_client.chat_with_structured_output.await_count == 2


@pytest.mark.asyncio
async def test_scenario_generator_retries_past_a_malformed_generation(sample_state, mock_llm_client):
    # A single malformed generation (model emits a broken scenarios array → raises) must not
    # kill the run; the loop retries and the next attempt succeeds.
    good = GeneratedScenarios(scenarios=[
        GeneratedScenario(id="p1", type="positive", description="book", turns=["Hi"]),
        GeneratedScenario(id="n1", type="negative", description="refuse", turns=["bad"]),
    ])
    mock_llm_client.chat_with_structured_output = AsyncMock(
        side_effect=[ValueError("malformed scenarios array"), good]
    )

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "scenarios_generated"
    assert {s.id for s in result["scenarios"]} == {"p1", "n1"}
    assert mock_llm_client.chat_with_structured_output.await_count == 2


@pytest.mark.asyncio
async def test_scenario_generator_all_attempts_malformed_fails_with_detail(sample_state, mock_llm_client):
    mock_llm_client.chat_with_structured_output = AsyncMock(side_effect=ValueError("boom"))

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "failed"
    assert "no scenarios" in result["error"].lower()
    assert "boom" in result["error"]


@pytest.mark.asyncio
async def test_scenario_generator_zero_scenarios_fails(sample_state, mock_llm_client):
    # Nothing at all to run is the one true failure.
    mock_llm_client.chat_with_structured_output.return_value = GeneratedScenarios(scenarios=[])

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "failed"
    assert "no scenarios" in result["error"].lower()


@pytest.mark.asyncio
async def test_scenario_generator_invalid_placeholder(sample_state, mock_llm_client):
    mock_llm_client.chat_with_structured_output.return_value = GeneratedScenarios(
        scenarios=[
            GeneratedScenario(
                id="positive-1",
                type="positive",
                description="Test",
                turns=["{{invalid/path}}"],
            ),
            GeneratedScenario(
                id="negative-1",
                type="negative",
                description="Test negative",
                turns=["Hello"],
            ),
        ]
    )

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "failed"
    assert "invalid placeholders" in result["error"]


@pytest.mark.asyncio
async def test_scenario_generator_llm_error(sample_state, mock_llm_client):
    mock_llm_client.chat_with_structured_output.side_effect = Exception("LLM error")

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "failed"
    assert "LLM error" in result["error"]


@pytest.mark.asyncio
async def test_scenario_generator_with_probes(sample_state, mock_llm_client):
    mock_llm_client.chat_with_structured_output.return_value = GeneratedScenarios(
        scenarios=[
            GeneratedScenario(
                id="positive-booking",
                type="positive",
                description="Book a slot",
                turns=["Book slot"],
                assertions=[
                    GeneratedAssertion(
                        type="world_state",
                        description="Slot removed",
                        path="/catalog/slots/0",
                        operator="removed",
                    )
                ],
            ),
            GeneratedScenario(
                id="probe-booking",
                type="positive",
                description="Verify slot is gone",
                turns=["Book same slot"],
                depends_on="positive-booking",
                depends_on_type="probe",
            ),
            GeneratedScenario(
                id="negative-unverified",
                type="negative",
                description="Unverified user",
                turns=["Book"],
            ),
        ]
    )

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "scenarios_generated"
    assert len(result["scenarios"]) == 3

    probe_scenario = next(s for s in result["scenarios"] if s.id == "probe-booking")
    assert probe_scenario.depends_on == "positive-booking"


@pytest.mark.asyncio
async def test_scenario_generator_target_constraint_passthrough(sample_state, mock_llm_client):
    mock_llm_client.chat_with_structured_output.return_value = GeneratedScenarios(
        scenarios=[
            GeneratedScenario(
                id="p1",
                type="positive",
                description="Valid booking",
                turns=["Book"],
            ),
            GeneratedScenario(
                id="n1",
                type="negative",
                description="Unverified user",
                turns=["Book"],
                target_constraint="only verified users can book",
            ),
        ]
    )

    result = await scenario_generator_node(sample_state, mock_llm_client)
    assert result["status"] == "scenarios_generated"

    positive = next(s for s in result["scenarios"] if s.id == "p1")
    negative = next(s for s in result["scenarios"] if s.id == "n1")
    assert positive.target_constraint is None
    assert negative.target_constraint == "only verified users can book"


@pytest.mark.asyncio
async def test_scenario_generator_agent_class(sample_state, mock_llm_client):
    mock_llm_client.chat_with_structured_output.return_value = GeneratedScenarios(
        scenarios=[
            GeneratedScenario(
                id="p1",
                type="positive",
                description="Test",
                turns=["Hi"],
            ),
            GeneratedScenario(
                id="n1",
                type="negative",
                description="Test",
                turns=["Hi"],
            ),
        ]
    )

    agent = ScenarioGeneratorAgent(mock_llm_client)
    result = await agent(sample_state)

    assert result["status"] == "scenarios_generated"
