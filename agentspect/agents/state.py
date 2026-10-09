from typing import Any, TypedDict

from agentspect.assets.base import Asset
from agentspect.models.suite import Scenario, ScenarioVerdict
from agentspect.models.transcript import Transcript
from agentspect.models.world_state import WorldState
from agentspect.models.yaml_input import AgentUnderTest, LLMConfig, TestConfig, WorldStateInput


class ATAGraphState(TypedDict, total=False):
    agent_under_test: AgentUnderTest
    world_state_input: WorldStateInput
    test_config: TestConfig
    llm_config: LLMConfig

    assets: list[Asset]

    world_state: WorldState
    scenarios: list[Scenario]
    scenario_batches: list[list[str]]

    current_batch_index: int
    current_batch_scenarios: list[Scenario]

    transcripts: dict[str, Transcript]
    verdicts: dict[str, ScenarioVerdict]
    patch_ops: dict[str, list[dict[str, Any]]]
    world_state_snapshots: dict[str, dict[str, Any]]

    skipped_scenarios: set[str]
    patch_failed_scenario_ids: list[str]
    patch_failed_reasons: dict[str, str]

    report: dict[str, Any]

    error: str | None
    status: str
