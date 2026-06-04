import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.agents.orchestrator import (
    OrchestratorAgent,
    build_graph,
    _initialize_state,
    _prepare_batch_node,
    _advance_batch_node,
    _should_continue_after_generate,
    _should_continue_after_prepare,
)
from backend.agents.state import ATAGraphState
from backend.models.suite import Scenario, ScenarioType
from backend.models.world_state import WorldState
from backend.models.yaml_input import (
    AgentUnderTest,
    LLMConfig,
    TestConfig,
    WorldStateInput,
    YAMLInput,
)


@pytest.fixture
def sample_yaml_input():
    return YAMLInput(
        agent_under_test=AgentUnderTest(
            name="Test Agent",
            url="http://localhost:8000",
            protocol="http",
            description="A test agent",
            capabilities=["booking"],
            known_limitations=["no rescheduling"],
        ),
        world_state=WorldStateInput(
            entities=[{"id": "user_1", "phone": "+1234567890"}],
            catalog={"slots": ["2026-05-20T10:00"]},
            constraints=["verified users only"],
            context={"language": "en"},
        ),
        test_config=TestConfig(total=2, positive=1, negative=1),
        llm_config=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
    )


@pytest.fixture
def valid_yaml():
    return """
agent_under_test:
  name: "Test Agent"
  url: "http://localhost:8080/chat"
  protocol: http
  description: "A test agent"
  capabilities:
    - booking
  known_limitations: []

world_state:
  entities:
    - id: customer_1
      phone: "+33612345678"
  catalog:
    available_slots:
      - "2026-05-20T10:00"
  constraints:
    - "only verified entities can book"
  context:
    current_time: "2026-05-16T08:00"

test_config:
  total: 2
  positive: 1
  negative: 1

llm_config:
  provider: anthropic
  model: claude-sonnet-4-20250514
"""


class TestInitializeState:
    def test_initialize_state_from_yaml_input(self, sample_yaml_input):
        state = _initialize_state(sample_yaml_input)

        assert state["agent_under_test"].name == "Test Agent"
        assert state["test_config"].total == 2
        assert state["llm_config"].provider == "anthropic"
        assert isinstance(state["world_state"], WorldState)
        assert state["scenarios"] == []
        assert state["current_batch_index"] == 0
        assert state["status"] == "initialized"
        assert state["error"] is None

    def test_initialize_state_world_state_data(self, sample_yaml_input):
        state = _initialize_state(sample_yaml_input)

        ws = state["world_state"]
        assert ws.data["entities"][0]["id"] == "user_1"
        assert "slots" in ws.data["catalog"]


class TestPrepareBatchNode:
    def test_prepare_first_batch(self):
        scenarios = [
            Scenario(id="s1", type=ScenarioType.POSITIVE, description="Test 1", turns=["Hi"]),
            Scenario(id="s2", type=ScenarioType.NEGATIVE, description="Test 2", turns=["Hello"]),
        ]

        state = ATAGraphState(
            scenarios=scenarios,
            scenario_batches=[["s1", "s2"]],
            current_batch_index=0,
            skipped_scenarios=set(),
        )

        result = _prepare_batch_node(state)

        assert len(result["current_batch_scenarios"]) == 2
        assert result["status"] == "batch_0_prepared"

    def test_prepare_batch_with_skipped(self):
        scenarios = [
            Scenario(id="s1", type=ScenarioType.POSITIVE, description="Test 1", turns=["Hi"]),
            Scenario(id="s2", type=ScenarioType.NEGATIVE, description="Test 2", turns=["Hello"]),
        ]

        state = ATAGraphState(
            scenarios=scenarios,
            scenario_batches=[["s1", "s2"]],
            current_batch_index=0,
            skipped_scenarios={"s1"},
        )

        result = _prepare_batch_node(state)

        assert len(result["current_batch_scenarios"]) == 1
        assert result["current_batch_scenarios"][0].id == "s2"

    def test_prepare_batch_all_complete(self):
        state = ATAGraphState(
            scenarios=[],
            scenario_batches=[["s1"]],
            current_batch_index=1,
            skipped_scenarios=set(),
        )

        result = _prepare_batch_node(state)

        assert result["current_batch_scenarios"] == []
        assert result["status"] == "all_batches_complete"


class TestAdvanceBatchNode:
    def test_advance_batch_increments_index(self):
        state = ATAGraphState(current_batch_index=0)
        result = _advance_batch_node(state)

        assert result["current_batch_index"] == 1
        assert result["status"] == "batch_0_complete"

    def test_advance_multiple_batches(self):
        state = ATAGraphState(current_batch_index=5)
        result = _advance_batch_node(state)

        assert result["current_batch_index"] == 6


class TestConditionalEdges:
    def test_should_continue_after_generate_success(self):
        state = ATAGraphState(status="scenarios_generated", error=None)
        result = _should_continue_after_generate(state)
        assert result == "prepare_batch"

    def test_should_continue_after_generate_error(self):
        state = ATAGraphState(status="failed", error="Some error")
        result = _should_continue_after_generate(state)
        assert result == "end"

    def test_should_continue_after_generate_status_failed(self):
        state = ATAGraphState(status="failed", error=None)
        result = _should_continue_after_generate(state)
        assert result == "end"

    def test_should_continue_after_prepare_more_batches(self):
        state = ATAGraphState(status="batch_0_prepared")
        result = _should_continue_after_prepare(state)
        assert result == "user_simulator"

    def test_should_continue_after_prepare_all_complete(self):
        state = ATAGraphState(status="all_batches_complete")
        result = _should_continue_after_prepare(state)
        assert result == "reporter"



class TestBuildGraph:
    def test_build_graph_has_all_nodes(self):
        mock_llm = MagicMock()
        mock_adapter = MagicMock()

        graph = build_graph(mock_llm, mock_adapter)

        node_names = list(graph.nodes.keys())
        assert "generate_scenarios" in node_names
        assert "prepare_batch" in node_names
        assert "user_simulator" in node_names
        assert "scorer" in node_names
        assert "world_state_patcher" in node_names
        assert "advance_batch" in node_names
        assert "reporter" in node_names


class TestOrchestratorAgent:
    @pytest.mark.asyncio
    async def test_orchestrator_initialize(self, valid_yaml):
        with patch("backend.agents.orchestrator.create_llm_client") as mock_create_llm, \
             patch("backend.agents.orchestrator.create_adapter") as mock_create_adapter:

            mock_create_llm.return_value = MagicMock()
            mock_create_adapter.return_value = MagicMock()

            orchestrator = OrchestratorAgent(valid_yaml)
            state = await orchestrator.initialize()

            assert state["agent_under_test"].name == "Test Agent"
            assert state["test_config"].total == 2
            mock_create_llm.assert_called_once()
            mock_create_adapter.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_initialize_invalid_yaml(self):
        invalid_yaml = "not: valid: yaml: [["
        orchestrator = OrchestratorAgent(invalid_yaml)

        with pytest.raises(Exception) as exc_info:
            await orchestrator.initialize()

        assert "YAML" in str(exc_info.value) or "syntax" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_orchestrator_progress_callback(self, valid_yaml):
        progress_events = []

        def callback(event, data):
            progress_events.append((event, data))

        with patch("backend.agents.orchestrator.create_llm_client") as mock_create_llm, \
             patch("backend.agents.orchestrator.create_adapter") as mock_create_adapter:

            mock_create_llm.return_value = MagicMock()
            mock_create_adapter.return_value = MagicMock()

            orchestrator = OrchestratorAgent(valid_yaml, progress_callback=callback)
            await orchestrator.initialize()

        event_names = [e[0] for e in progress_events]
        assert "parsing_yaml" in event_names
        assert "yaml_validated" in event_names
        assert "initialized" in event_names

    @pytest.mark.asyncio
    async def test_orchestrator_missing_required_keys(self):
        yaml_missing_test_config = """
agent_under_test:
  name: "Test Agent"
  url: "http://localhost:8080"
  protocol: http
  description: "Test"

world_state:
  entities: []

llm_config:
  provider: anthropic
  model: test
"""
        orchestrator = OrchestratorAgent(yaml_missing_test_config)

        with pytest.raises(Exception) as exc_info:
            await orchestrator.initialize()

        assert "test_config" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_orchestrator_rag_key_rejected(self):
        yaml_with_rag = """
agent_under_test:
  name: "Test Agent"
  url: "http://localhost:8080"
  protocol: http
  description: "Test"

world_state:
  entities: []

test_config:
  total: 1
  positive: 1
  negative: 0

llm_config:
  provider: anthropic
  model: test

rag:
  enabled: true
"""
        orchestrator = OrchestratorAgent(yaml_with_rag)

        with pytest.raises(Exception) as exc_info:
            await orchestrator.initialize()

        assert "rag" in str(exc_info.value).lower()
