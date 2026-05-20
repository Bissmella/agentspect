import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.agents.scenario_generator import (
    ScenarioGeneratorAgent,
    scenario_generator_node,
    GeneratedScenarios,
    GeneratedScenario,
    GeneratedAssertion,
)
from backend.agents.state import ATAGraphState
from backend.models.suite import ScenarioType
from backend.models.yaml_input import AgentUnderTest, LLMConfig, TestConfig, WorldStateInput


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
async def test_scenario_generator_count_mismatch(sample_state, mock_llm_client):
    mock_llm_client.chat_with_structured_output.return_value = GeneratedScenarios(
        scenarios=[
            GeneratedScenario(
                id="positive-1",
                type="positive",
                description="Test",
                turns=["Hello"],
            ),
        ]
    )

    result = await scenario_generator_node(sample_state, mock_llm_client)

    assert result["status"] == "failed"
    assert "negative scenarios" in result["error"]


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
