from typing import Any, TypedDict

from backend.models.suite import Scenario, ScenarioVerdict
from backend.models.transcript import Transcript
from backend.models.world_state import WorldState
from backend.models.yaml_input import AgentUnderTest, LLMConfig, TestConfig, WorldStateInput


class ATAGraphState(TypedDict, total=False):
    agent_under_test: AgentUnderTest
    world_state_input: WorldStateInput
    test_config: TestConfig
    llm_config: LLMConfig

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
    patch_failed: bool
    patch_failed_scenario_id: str | None
    patch_failed_reason: str | None

    report: dict[str, Any]

    error: str | None
    status: str
