from typing import Any, Callable, Literal

from langgraph.graph import END, StateGraph

from ata.adapters.base import ProtocolAdapter
from ata.adapters.callable_adapter import AgentCallable, CallableAdapter
from ata.adapters.ws_adapter import create_adapter
from ata.agents.asset_ingestion import asset_ingestion_node
from ata.agents.reporter import reporter_node
from ata.agents.scenario_generator import scenario_generator_node
from ata.agents.scorer import scorer_node
from ata.agents.state import ATAGraphState
from ata.agents.user_simulator import user_simulator_node
from ata.agents.world_state_patcher import world_state_patcher_node
from ata.llm.client import LLMClient, create_llm_client
from ata.models.world_state import WorldState
from ata.models.yaml_input import YAMLInput
from ata.services.yaml_parser import YAMLValidationError, parse_and_validate


ProgressCallback = Callable[[str, dict[str, Any]], None]


def _initialize_state(
    yaml_input: YAMLInput,
    assets: list | None = None,
) -> ATAGraphState:
    world_state = WorldState.from_dict(yaml_input.world_state.model_dump())

    return ATAGraphState(
        agent_under_test=yaml_input.agent_under_test,
        world_state_input=yaml_input.world_state,
        test_config=yaml_input.test_config,
        llm_config=yaml_input.llm_config,
        assets=assets or [],
        world_state=world_state,
        scenarios=[],
        scenario_batches=[],
        current_batch_index=0,
        current_batch_scenarios=[],
        transcripts={},
        verdicts={},
        patch_ops={},
        world_state_snapshots={},
        skipped_scenarios=set(),
        patch_failed_scenario_ids=[],
        patch_failed_reasons={},
        report={},
        error=None,
        status="initialized",
    )


def _create_asset_ingestion_node(llm_client: LLMClient):
    async def node(state: ATAGraphState) -> dict[str, Any]:
        return await asset_ingestion_node(state, llm_client)

    return node


def _create_scenario_generator_node(llm_client: LLMClient):
    async def node(state: ATAGraphState) -> dict[str, Any]:
        return await scenario_generator_node(state, llm_client)

    return node


def _create_user_simulator_node(llm_client: LLMClient, adapter: ProtocolAdapter):
    async def node(state: ATAGraphState) -> dict[str, Any]:
        return await user_simulator_node(state, llm_client, adapter)

    return node


def _create_scorer_node(llm_client: LLMClient):
    async def node(state: ATAGraphState) -> dict[str, Any]:
        return await scorer_node(state, llm_client)

    return node


def _create_world_state_patcher_node(llm_client: LLMClient):
    async def node(state: ATAGraphState) -> dict[str, Any]:
        return await world_state_patcher_node(state, llm_client)

    return node


def _create_reporter_node(llm_client: LLMClient):
    async def node(state: ATAGraphState) -> dict[str, Any]:
        return await reporter_node(state, llm_client)

    return node


def _prepare_batch_node(state: ATAGraphState) -> dict[str, Any]:
    batches = state.get("scenario_batches", [])
    current_index = state.get("current_batch_index", 0)
    all_scenarios = state.get("scenarios", [])
    skipped = state.get("skipped_scenarios", set())

    if current_index >= len(batches):
        return {
            "current_batch_scenarios": [],
            "status": "all_batches_complete",
        }

    batch_ids = batches[current_index]
    scenarios_map = {s.id: s for s in all_scenarios}

    batch_scenarios = []
    for sid in batch_ids:
        if sid not in skipped and sid in scenarios_map:
            batch_scenarios.append(scenarios_map[sid])

    return {
        "current_batch_scenarios": batch_scenarios,
        "status": f"batch_{current_index}_prepared",
    }


def _advance_batch_node(state: ATAGraphState) -> dict[str, Any]:
    current_index = state.get("current_batch_index", 0)
    return {
        "current_batch_index": current_index + 1,
        "status": f"batch_{current_index}_complete",
    }


def _should_continue_after_generate(
    state: ATAGraphState,
) -> Literal["prepare_batch", "end"]:
    if state.get("error") or state.get("status") == "failed":
        return "end"
    return "prepare_batch"


def _should_continue_after_prepare(
    state: ATAGraphState,
) -> Literal["user_simulator", "reporter"]:
    if state.get("status") == "all_batches_complete":
        return "reporter"
    return "user_simulator"


def build_graph(
    llm_client: LLMClient,
    adapter: ProtocolAdapter,
) -> StateGraph:
    graph = StateGraph(ATAGraphState)

    graph.add_node("asset_ingestion", _create_asset_ingestion_node(llm_client))
    graph.add_node("generate_scenarios", _create_scenario_generator_node(llm_client))
    graph.add_node("prepare_batch", _prepare_batch_node)
    graph.add_node("user_simulator", _create_user_simulator_node(llm_client, adapter))
    graph.add_node("scorer", _create_scorer_node(llm_client))
    graph.add_node("world_state_patcher", _create_world_state_patcher_node(llm_client))
    graph.add_node("advance_batch", _advance_batch_node)
    graph.add_node("reporter", _create_reporter_node(llm_client))

    graph.set_entry_point("asset_ingestion")
    graph.add_edge("asset_ingestion", "generate_scenarios")

    graph.add_conditional_edges(
        "generate_scenarios",
        _should_continue_after_generate,
        {
            "prepare_batch": "prepare_batch",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "prepare_batch",
        _should_continue_after_prepare,
        {
            "user_simulator": "user_simulator",
            "reporter": "reporter",
        },
    )

    graph.add_edge("user_simulator", "scorer")
    graph.add_edge("scorer", "world_state_patcher")

    graph.add_edge("world_state_patcher", "advance_batch")

    graph.add_edge("advance_batch", "prepare_batch")
    graph.add_edge("reporter", END)

    return graph


class OrchestratorAgent:
    def __init__(
        self,
        yaml_str: str,
        progress_callback: ProgressCallback | None = None,
        agent: AgentCallable | None = None,
        base_dir: str | None = None,
    ):
        self.yaml_str = yaml_str
        self.progress_callback = progress_callback
        self._agent = agent
        self._base_dir = base_dir
        self._graph = None
        self._llm_client: LLMClient | None = None
        self._adapter: ProtocolAdapter | None = None

    def _emit_progress(self, event: str, data: dict[str, Any] | None = None) -> None:
        if self.progress_callback:
            payload = {
                "event": event,
                "status": None,
                "error": None,
                "node": None,
            }
            if data:
                payload.update(data)
            self.progress_callback(event, payload)

    async def _close_adapter(self) -> None:
        if self._adapter:
            try:
                await self._adapter.close()
            except Exception:
                pass

    async def initialize(self) -> ATAGraphState:
        self._emit_progress("parsing_yaml")

        try:
            yaml_input, yaml_hash = parse_and_validate(self.yaml_str)
        except YAMLValidationError as e:
            self._emit_progress("yaml_validation_failed", {"error": str(e)})
            raise

        self._emit_progress("yaml_validated", {"hash": yaml_hash})

        llm_config = yaml_input.llm_config
        try:
            self._llm_client = create_llm_client(
                provider=llm_config.provider,
                model=llm_config.model,
            )
        except Exception as e:
            self._emit_progress("initialization_failed", {
                "error": f"LLM client creation failed: {e}",
            })
            raise RuntimeError(f"Failed to create LLM client for provider '{llm_config.provider}': {e}") from e

        agent_config = yaml_input.agent_under_test
        use_callable = self._agent is not None or agent_config.protocol == "callable"
        try:
            if use_callable:
                if self._agent is None:
                    raise ValueError(
                        "protocol 'callable' requires an `agent` callable passed to "
                        "run_suite()/OrchestratorAgent()"
                    )
                self._adapter = CallableAdapter(self._agent)
            else:
                self._adapter = create_adapter(
                    protocol=agent_config.protocol,
                    url=agent_config.url,
                    voice=yaml_input.voice,
                )
        except Exception as e:
            self._llm_client = None
            self._emit_progress("initialization_failed", {
                "error": f"Protocol adapter creation failed: {e}",
            })
            target = "callable agent" if use_callable else f"{agent_config.protocol} adapter for '{agent_config.url}'"
            raise RuntimeError(f"Failed to create {target}: {e}") from e

        assets = []
        if yaml_input.assets:
            from ata.assets.service import load_assets

            try:
                assets = load_assets(yaml_input.assets, self._base_dir)
            except Exception as e:
                self._emit_progress("initialization_failed", {
                    "error": f"Asset loading failed: {e}",
                })
                raise RuntimeError(f"Failed to load assets: {e}") from e
            self._emit_progress("assets_loaded", {"count": len(assets)})

        self._graph = build_graph(self._llm_client, self._adapter)

        initial_state = _initialize_state(yaml_input, assets)
        self._emit_progress("initialized", {
            "agent_name": agent_config.name,
            "total_scenarios": yaml_input.test_config.total,
        })

        return initial_state

    async def run(self) -> dict[str, Any]:
        initial_state = await self.initialize()

        self._emit_progress("starting_execution")

        compiled = self._graph.compile()

        try:
            final_state = None
            async for event in compiled.astream(initial_state):
                for node_name, node_output in event.items():
                    self._emit_progress(f"node_{node_name}", {
                        "node": node_name,
                        "status": node_output.get("status"),
                        "error": node_output.get("error"),
                    })

                    if node_name == "reporter":
                        final_state = node_output
        finally:
            await self._close_adapter()

        if not final_state or "report" not in final_state:
            error_msg = initial_state.get("error") or "Reporter node never executed"
            self._emit_progress("completed", {"error": error_msg})
            raise RuntimeError(f"Suite run did not produce a report: {error_msg}")

        report = final_state["report"]
        self._emit_progress("completed", {
            "verdict_counts": report.get("verdict_counts", {}),
        })

        return report

    async def run_from_state(self, state: ATAGraphState) -> dict[str, Any]:
        if not self._graph:
            raise RuntimeError("Graph not initialized. Call initialize() first.")

        compiled = self._graph.compile()

        try:
            final_state = None
            async for event in compiled.astream(state):
                for node_name, node_output in event.items():
                    self._emit_progress(f"node_{node_name}", {
                        "node": node_name,
                        "status": node_output.get("status"),
                        "error": node_output.get("error"),
                    })

                    if node_name == "reporter":
                        final_state = node_output
        finally:
            await self._close_adapter()

        if not final_state or "report" not in final_state:
            raise RuntimeError("Suite run did not produce a report")

        return final_state["report"]


async def run_suite(
    yaml_str: str,
    progress_callback: ProgressCallback | None = None,
    agent: AgentCallable | None = None,
    base_dir: str | None = None,
) -> dict[str, Any]:
    """Run a full test suite from a YAML config string.

    Pass ``agent`` to test an in-process Python callable directly (sync or
    async, ``(message)`` or ``(message, history)``) instead of a live HTTP/
    WebSocket endpoint. When ``agent`` is given, the YAML's ``url``/``protocol``
    are not required to describe a network endpoint.

    ``base_dir`` resolves relative ``assets`` paths in the YAML (default: cwd).
    """
    orchestrator = OrchestratorAgent(yaml_str, progress_callback, agent=agent, base_dir=base_dir)
    return await orchestrator.run()
