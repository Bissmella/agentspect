from typing import Any

from pydantic import BaseModel, Field

from backend.agents.state import ATAGraphState
from backend.llm.client import LLMClient
from backend.models.suite import (
    Assertion,
    BehavioralAssertion,
    DependsOnType,
    Scenario,
    ScenarioType,
    TranscriptAssertion,
    WorldStateAssertion,
)
from backend.services.dag import build_dag, topological_batches
from backend.services.placeholder import validate_placeholders


class GeneratedAssertion(BaseModel):
    type: str = Field(description="One of: world_state, transcript, behavioral")
    description: str
    path: str | None = None
    operator: str | None = None
    expected_value: Any | None = None
    check: str | None = None
    speaker: str | None = None
    expected_behavior: str | None = None


class GeneratedScenario(BaseModel):
    id: str
    type: str = Field(description="positive or negative")
    description: str
    turns: list[str]
    persona: str | None = None
    assertions: list[GeneratedAssertion] = Field(default_factory=list)
    depends_on: str | None = None
    depends_on_type: str | None = None


class GeneratedScenarios(BaseModel):
    scenarios: list[GeneratedScenario]


def _convert_assertion(gen_assertion: GeneratedAssertion) -> Assertion:
    if gen_assertion.type == "world_state":
        return WorldStateAssertion(
            description=gen_assertion.description,
            path=gen_assertion.path or "",
            operator=gen_assertion.operator or "equals",
            expected_value=gen_assertion.expected_value,
        )
    elif gen_assertion.type == "transcript":
        return TranscriptAssertion(
            description=gen_assertion.description,
            check=gen_assertion.check or "",
            speaker=gen_assertion.speaker or "agent",
        )
    elif gen_assertion.type == "behavioral":
        return BehavioralAssertion(
            description=gen_assertion.description,
            expected_behavior=gen_assertion.expected_behavior or "confirmation",
        )
    else:
        raise ValueError(f"Unknown assertion type: {gen_assertion.type}")


def _convert_scenario(gen_scenario: GeneratedScenario) -> Scenario:
    assertions = [_convert_assertion(a) for a in gen_scenario.assertions]

    depends_on_type = None
    if gen_scenario.depends_on_type:
        depends_on_type = DependsOnType(gen_scenario.depends_on_type)

    return Scenario(
        id=gen_scenario.id,
        type=ScenarioType(gen_scenario.type),
        description=gen_scenario.description,
        turns=gen_scenario.turns,
        persona=gen_scenario.persona,
        assertions=assertions,
        depends_on=gen_scenario.depends_on,
        depends_on_type=depends_on_type,
    )


def _build_system_prompt(state: ATAGraphState) -> str:
    agent = state["agent_under_test"]
    world_state = state["world_state_input"]
    test_config = state["test_config"]

#TODO to be enhanced
    return f"""You are a scenario generator for an agent testing framework.

## Agent Under Test
Name: {agent.name}
Description: {agent.description}
Capabilities: {', '.join(agent.capabilities)}
Known Limitations: {', '.join(agent.known_limitations)}

## World State
Entities: {world_state.entities}
Catalog: {world_state.catalog}
Constraints: {world_state.constraints}
Context: {world_state.context}

## Requirements
Generate exactly {test_config.total} test scenarios:
- {test_config.positive} positive scenarios (agent should succeed)
- {test_config.negative} negative scenarios (agent should refuse/fail)

## Rules for scenario generation

1. POSITIVE scenarios test that the agent correctly handles valid requests within constraints.
   - Use valid entities from world_state.entities
   - Use valid values from world_state.catalog
   - Respect all constraints

2. NEGATIVE scenarios test that the agent correctly refuses invalid requests.
   - Deliberately violate ONE constraint per scenario
   - Use slightly mutated values (wrong phone digit, out-of-range time, unknown entity)
   - Do NOT generate arbitrary nonsense - cross boundaries deliberately

3. Do NOT generate scenarios for known_limitations - those are out of scope.

4. Placeholders: Use {{{{path/to/value}}}} syntax to reference world_state values.
   Example: {{{{entities/0/phone}}}} resolves to the first entity's phone number.

5. For POSITIVE scenarios with stateful side effects:
   - Generate a probe scenario with depends_on pointing to the primary scenario
   - Set depends_on_type to "probe"
   - Probe should attempt the same action again to verify state was updated

6. For NEGATIVE scenarios that could corrupt state if agent misbehaves:
   - Generate a defensive_probe scenario with depends_on
   - Set depends_on_type to "defensive_probe"
   - Defensive probe verifies no state corruption occurred

7. Assertions:
   - world_state assertions: path (JSON pointer), operator (removed|added|equals|contains|not_contains)
   - transcript assertions: semantic check evaluated by LLM
   - behavioral assertions: expected_behavior (refusal|confirmation|clarification|escalation)

8. Each scenario ID must be unique. Use descriptive kebab-case IDs.

9. Turns are user messages in sequence. Keep them realistic for the persona."""


def _build_user_prompt() -> str:
    return "Generate the test scenarios now. Return them as a structured output with a 'scenarios' array."


async def scenario_generator_node(
    state: ATAGraphState, llm_client: LLMClient
) -> dict[str, Any]:
    system_prompt = _build_system_prompt(state)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_user_prompt()},
    ]

    try:
        result = await llm_client.chat_with_structured_output(
            messages, GeneratedScenarios, temperature=0.7
        )

        scenarios = [_convert_scenario(s) for s in result.scenarios]

        test_config = state["test_config"]
        positive_count = sum(1 for s in scenarios if s.type == ScenarioType.POSITIVE)
        negative_count = sum(1 for s in scenarios if s.type == ScenarioType.NEGATIVE)

        if positive_count < test_config.positive:
            return {
                "error": f"Generated {positive_count} positive scenarios, expected {test_config.positive}",
                "status": "failed",
            }
        if negative_count < test_config.negative:
            return {
                "error": f"Generated {negative_count} negative scenarios, expected {test_config.negative}",
                "status": "failed",
            }

        world_state_data = state["world_state_input"].model_dump()
        for scenario in scenarios:
            for turn in scenario.turns:
                errors = validate_placeholders(turn, world_state_data)
                if errors:
                    return {
                        "error": f"Scenario '{scenario.id}' has invalid placeholders: {errors}",
                        "status": "failed",
                    }

        dag = build_dag(scenarios)
        batches = topological_batches(scenarios)

        return {
            "scenarios": scenarios,
            "scenario_batches": batches,
            "status": "scenarios_generated",
        }

    except Exception as e:
        return {
            "error": f"Scenario generation failed: {str(e)}",
            "status": "failed",
        }


class ScenarioGeneratorAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def __call__(self, state: ATAGraphState) -> dict[str, Any]:
        return await scenario_generator_node(state, self.llm_client)
