import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentspect.agents.orchestrator import (
    OrchestratorAgent,
    build_graph,
    _initialize_state,
    _prepare_batch_node,
    _advance_batch_node,
    _should_continue_after_generate,
    _should_continue_after_prepare,
)
from agentspect.agents.state import ATAGraphState
from agentspect.models.suite import Scenario, ScenarioType
from agentspect.models.world_state import WorldState
from agentspect.models.yaml_input import (
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

    def test_prepare_batch_skips_meaningless_probes(self):
        from agentspect.models.suite import DependsOnType, Verdict, ScenarioVerdict
        scenarios = [
            Scenario(id="p", type=ScenarioType.POSITIVE, description="book", turns=["Hi"]),
            Scenario(id="probe-ok", type=ScenarioType.POSITIVE, description="probe", turns=["Hi"],
                     depends_on="p", depends_on_type=DependsOnType.PROBE),
            Scenario(id="n", type=ScenarioType.NEGATIVE, description="neg", turns=["Hi"]),
            Scenario(id="dprobe", type=ScenarioType.NEGATIVE, description="dprobe", turns=["Hi"],
                     depends_on="n", depends_on_type=DependsOnType.DEFENSIVE_PROBE),
        ]
        # p FAILED → its verification probe is meaningless; n did NOT fail → its defensive
        # probe is meaningless. Both must be skipped.
        state = ATAGraphState(
            scenarios=scenarios,
            scenario_batches=[["probe-ok", "dprobe"]],
            current_batch_index=0,
            skipped_scenarios=set(),
            verdicts={
                "p": ScenarioVerdict(scenario_id="p", verdict=Verdict.FAILURE, reason=""),
                "n": ScenarioVerdict(scenario_id="n", verdict=Verdict.SUCCESS, reason=""),
            },
        )
        result = _prepare_batch_node(state)
        assert result["current_batch_scenarios"] == []

    def test_prepare_batch_runs_probes_for_the_right_parent_outcome(self):
        from agentspect.models.suite import DependsOnType, Verdict, ScenarioVerdict
        scenarios = [
            Scenario(id="p", type=ScenarioType.POSITIVE, description="book", turns=["Hi"]),
            Scenario(id="probe-ok", type=ScenarioType.POSITIVE, description="probe", turns=["Hi"],
                     depends_on="p", depends_on_type=DependsOnType.PROBE),
            Scenario(id="n", type=ScenarioType.NEGATIVE, description="neg", turns=["Hi"]),
            Scenario(id="dprobe", type=ScenarioType.NEGATIVE, description="dprobe", turns=["Hi"],
                     depends_on="n", depends_on_type=DependsOnType.DEFENSIVE_PROBE),
        ]
        state = ATAGraphState(
            scenarios=scenarios,
            scenario_batches=[["probe-ok", "dprobe"]],
            current_batch_index=0,
            skipped_scenarios=set(),
            verdicts={
                "p": ScenarioVerdict(scenario_id="p", verdict=Verdict.SUCCESS_UNVERIFIED, reason=""),
                "n": ScenarioVerdict(scenario_id="n", verdict=Verdict.FAILURE, reason=""),
            },
        )
        result = _prepare_batch_node(state)
        assert {s.id for s in result["current_batch_scenarios"]} == {"probe-ok", "dprobe"}

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
        with patch("agentspect.agents.orchestrator.create_llm_client") as mock_create_llm, \
             patch("agentspect.agents.orchestrator.create_adapter") as mock_create_adapter:

            mock_create_llm.return_value = MagicMock()
            mock_create_adapter.return_value = MagicMock()

            orchestrator = OrchestratorAgent(valid_yaml)
            state = await orchestrator.initialize()

            assert state["agent_under_test"].name == "Test Agent"
            assert state["test_config"].total == 2
            mock_create_llm.assert_called_once()
            mock_create_adapter.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_injected_adapter_bypasses_create_adapter(self, valid_yaml):
        """A host-supplied adapter is used verbatim; the YAML-derived path is skipped."""
        injected = MagicMock()

        with patch("agentspect.agents.orchestrator.create_llm_client") as mock_create_llm, \
             patch("agentspect.agents.orchestrator.create_adapter") as mock_create_adapter:

            mock_create_llm.return_value = MagicMock()

            orchestrator = OrchestratorAgent(valid_yaml, adapter=injected)
            await orchestrator.initialize()

            assert orchestrator._adapter is injected
            mock_create_adapter.assert_not_called()

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

        with patch("agentspect.agents.orchestrator.create_llm_client") as mock_create_llm, \
             patch("agentspect.agents.orchestrator.create_adapter") as mock_create_adapter:

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


class TestGenerateExecuteSplit:
    def test_build_generate_graph_nodes(self):
        from agentspect.agents.orchestrator import build_generate_graph

        nodes = list(build_generate_graph(MagicMock()).nodes.keys())
        assert "asset_ingestion" in nodes and "generate_scenarios" in nodes
        assert "user_simulator" not in nodes and "reporter" not in nodes

    def test_build_execute_graph_nodes(self):
        from agentspect.agents.orchestrator import build_execute_graph

        nodes = list(build_execute_graph(MagicMock(), MagicMock()).nodes.keys())
        assert "prepare_batch" in nodes and "reporter" in nodes
        assert "generate_scenarios" not in nodes and "asset_ingestion" not in nodes

    def test_scenario_roundtrips_through_json(self):
        """The platform persists/edits scenarios as JSON, so this must survive."""
        from agentspect.models.suite import BehavioralAssertion, Scenario, ScenarioType

        sc = Scenario(
            id="pos-1",
            type=ScenarioType.POSITIVE,
            description="book a slot",
            turns=["Hi, book 2026-05-20T10:00"],
            assertions=[BehavioralAssertion(description="confirms", expected_behavior="confirmation")],
        )
        back = Scenario.model_validate(sc.model_dump(mode="json"))
        assert back.id == "pos-1"
        assert back.type == ScenarioType.POSITIVE
        assert isinstance(back.assertions[0], BehavioralAssertion)
        assert back.assertions[0].expected_behavior == "confirmation"

    @pytest.mark.asyncio
    async def test_generate_returns_serialized_scenarios(self, valid_yaml):
        from agentspect.agents.scenario_generator import (
            GeneratedAssertion,
            GeneratedScenario,
            GeneratedScenarios,
        )

        gen = GeneratedScenarios(scenarios=[
            GeneratedScenario(
                id="pos-1", type="positive", description="valid booking",
                turns=["Hi, I'd like to book a slot."],
                assertions=[GeneratedAssertion(type="behavioral", description="confirms", expected_behavior="confirmation")],
            ),
            GeneratedScenario(
                id="neg-1", type="negative", description="unverified caller",
                turns=["Book me in, I'm not registered."],
                target_constraint="only verified entities can book",
                assertions=[GeneratedAssertion(type="behavioral", description="refuses", expected_behavior="refusal")],
            ),
        ])
        mock_llm = MagicMock()
        mock_llm.chat_with_structured_output = AsyncMock(return_value=gen)

        with patch("agentspect.agents.orchestrator.create_llm_client", return_value=mock_llm), \
             patch("agentspect.agents.orchestrator.create_adapter", return_value=MagicMock()):
            scenarios = await OrchestratorAgent(valid_yaml).generate()

        assert [s["id"] for s in scenarios] == ["pos-1", "neg-1"]
        assert scenarios[0]["type"] == "positive"
        assert scenarios[0]["turns"] == ["Hi, I'd like to book a slot."]
        assert scenarios[1]["assertions"][0]["type"] == "behavioral"


def test_patcher_runs_before_scorer_so_world_state_assertions_have_snapshots():
    # The scorer evaluates world_state assertions against the before/after snapshots the
    # patcher writes. If the scorer ran first (the old order) every stateful assertion was
    # scored against empty snapshots and always MISSed. Guard the edge order.
    from agentspect.agents.orchestrator import build_execute_graph

    g = build_execute_graph(MagicMock(), MagicMock())
    edges = {(s, e) for s, e in g.edges}
    assert ("user_simulator", "world_state_patcher") in edges
    assert ("world_state_patcher", "scorer") in edges
    assert ("scorer", "advance_batch") in edges
    # the reverse (scorer before patcher) must be gone
    assert ("scorer", "world_state_patcher") not in edges
