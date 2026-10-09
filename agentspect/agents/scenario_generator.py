from typing import Any

from pydantic import BaseModel, Field

from agentspect.agents.state import ATAGraphState
from agentspect.llm.client import LLMClient
from agentspect.models.suite import (
    Assertion,
    BehavioralAssertion,
    DependsOnType,
    Scenario,
    ScenarioType,
    TranscriptAssertion,
    WorldStateAssertion,
)
from agentspect.services.dag import build_dag, topological_batches
from agentspect.services.placeholder import validate_placeholders


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
    target_constraint: str | None = None


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
        target_constraint=gen_scenario.target_constraint,
    )


def _build_system_prompt(state: ATAGraphState) -> str:
    agent = state["agent_under_test"]
    world_state = state["world_state_input"]
    test_config = state["test_config"]

    import json
    entities_json = json.dumps(world_state.entities, indent=2)
    catalog_json = json.dumps(world_state.catalog, indent=2)
    constraints_json = json.dumps(world_state.constraints, indent=2)
    context_json = json.dumps(world_state.context, indent=2)

    return f"""You are a test scenario generator for an AI agent testing framework.

## Agent Under Test
- Name: {agent.name}
- Description: {agent.description}
- Capabilities: {', '.join(agent.capabilities) if agent.capabilities else 'general conversation'}
- Known Limitations (DO NOT test these): {', '.join(agent.known_limitations) if agent.known_limitations else 'none'}

## World State Data

### Entities (users, objects the agent can look up)
{entities_json}

### Catalog (available options the agent can offer)
{catalog_json}

### Constraints (rules the agent must follow)
{constraints_json}

### Context (runtime facts)
{context_json}

## Your Task
Generate exactly {test_config.positive} POSITIVE and {test_config.negative} NEGATIVE test scenarios.

IMPORTANT: Probes are ADDITIONAL scenarios that verify state changes. They do NOT count toward the {test_config.positive}/{test_config.negative} totals.

## POSITIVE Scenarios (agent should succeed)
- Test valid requests that follow all constraints
- CRITICAL: User turns MUST use EXACT values from world_state
  - Use the exact phone numbers from entities (e.g., "+1111111111", not made-up numbers)
  - Use the exact slot times from catalog (e.g., "2026-05-20T10:00")
  - Use the exact entity names from entities
- If the scenario has a stateful side effect (booking, registration), add a PROBE scenario:
  - Set depends_on to the parent scenario ID
  - Set depends_on_type to "probe"
  - The probe attempts the same action again - it should FAIL because state changed

## NEGATIVE Scenarios (agent should refuse)
- Test invalid requests that violate a constraint or push adversarial boundaries
- Ways to create invalid scenarios:
  - Use an entity that doesn't exist (e.g., phone "+9999999999" not in entities)
  - Use a catalog value that doesn't exist (e.g., slot "2026-05-20T08:00" not in catalog)
  - Violate a constraint (e.g., unregistered user trying to book if constraint says "only registered users can book")
  - Broader adversarial inputs: out-of-scope requests, prompt injection attempts, unrecognized input formats
- For every negative scenario, set `target_constraint` to describe what boundary is being tested:
  - If violating a specific constraint from the list above, use the EXACT text of that constraint
  - If adversarial in a broader sense, use a short descriptive label like "out-of-scope request" or "prompt injection attempt"
- If the negative scenario could corrupt state, add a DEFENSIVE_PROBE:
  - Set depends_on_type to "defensive_probe"
  - Verifies state wasn't corrupted by the invalid request

## Turn Writing Rules
1. Each turn is a single user message (string)
2. Write realistic conversational messages, not commands
3. Include the persona's actual data in the message:
   - GOOD: "Hi, my phone number is +1111111111"
   - BAD: "Hi, my phone number is 123-456-7890" (made up)
4. Keep turns concise - 1-3 sentences each
5. Typical flow: greeting → provide identity → state request → confirm details

## Assertion Writing Rules

### world_state assertions (for stateful scenarios)
- path: JSON pointer to the value (e.g., "/catalog/slots/2026-05-20T10:00")
- operator: "removed" (value was deleted), "added" (value was created), "equals" (value matches expected), "contains", "not_contains"
- expected_value: the expected value (required for equals/contains/not_contains)
- Example: After booking slot X, use operator "removed" with path "/catalog/slots/X" OR operator "equals" with expected_value "booked"

### transcript assertions (for conversation checks)
- check: natural language condition (e.g., "agent confirmed the booking with a confirmation number")
- speaker: "agent" or "user"

### behavioral assertions (for behavior classification)
- expected_behavior: "confirmation" (agent confirms action), "refusal" (agent declines), "clarification" (agent asks for more info), "escalation" (agent escalates)

## Output Format
Return scenarios as a JSON array. Each scenario has:
- id: unique kebab-case identifier
- type: "positive" or "negative"
- description: what this test verifies
- turns: array of user message strings
- persona: entity ID if acting as a specific entity (optional)
- assertions: array of assertion objects
- depends_on: parent scenario ID (only for probes)
- depends_on_type: "probe" or "defensive_probe" (only when depends_on is set)
- target_constraint: exact constraint text or short descriptive label of what boundary is being tested (required for all negative scenarios, null for positive/probes)"""


def _build_user_prompt() -> str:
    return """Generate the test scenarios now.

Remember:
1. Use EXACT values from world_state (real phone numbers, real slot times, real entity names)
2. Probes are ADDITIONAL and don't count toward the positive/negative totals
3. Each positive scenario with side effects needs a probe
4. Each negative scenario that could corrupt state needs a defensive_probe

Return structured output with a 'scenarios' array."""


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
