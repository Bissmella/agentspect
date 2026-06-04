from typing import Any

from pydantic import BaseModel, Field

from backend.agents.state import ATAGraphState
from backend.llm.client import LLMClient
from backend.models.suite import ScenarioVerdict, Verdict
from backend.models.transcript import Transcript
from backend.models.world_state import PatchValidationError, WorldState
from backend.services.dag import build_dag, cascade_skip


class PatchOperation(BaseModel):
    op: str = Field(description="One of: add, remove, replace, move, copy, test")
    path: str = Field(description="JSON pointer path like /entities/0/verified")
    value: Any | None = None
    from_path: str | None = Field(None, alias="from")


class PatchWorldStateOutput(BaseModel):
    ops: list[PatchOperation] = Field(default_factory=list)
    reasoning: str = Field(description="Brief explanation of inferred mutations")


PATCH_TOOL = {
    "name": "patch_world_state",
    "description": "Apply mutations to world_state inferred from the transcript. "
                   "Only emit operations for state changes that actually occurred based on the conversation.",
    "parameters": {
        "type": "object",
        "properties": {
            "ops": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "enum": ["add", "remove", "replace", "move", "copy", "test"],
                        },
                        "path": {"type": "string"},
                        "value": {},
                        "from": {"type": "string"},
                    },
                    "required": ["op", "path"],
                },
            },
        },
        "required": ["ops"],
    },
}


def _build_patcher_prompt(
    transcript: Transcript, world_state: WorldState
) -> list[dict[str, str]]:
    import json

    transcript_text = "\n".join(
        f"Turn {i+1}:\n  User: {turn.user_message}\n  Agent: {turn.agent_response}"
        for i, turn in enumerate(transcript.turns)
    )

    world_state_json = json.dumps(world_state.to_dict(), indent=2)

    system_prompt = f"""You analyze conversation transcripts to determine what world_state changes occurred.

## Current World State
```json
{world_state_json}
```

## Your Task
1. Read the conversation transcript
2. Identify if the agent CONFIRMED completing any action (booking, registration, update, etc.)
3. Emit patch operations ONLY for confirmed, completed actions
4. If the agent refused, failed, or action was not completed → emit empty ops array

## Patch Operation Rules

### When to use each operation:
- **remove**: Delete a value (e.g., remove a booked slot from available slots)
  - Path must exist in current state
- **replace**: Update an existing value (e.g., change slot status from "available" to "booked")
  - Path must exist in current state
- **add**: Add a new value (e.g., add a new reservation to an array)
  - Parent path must exist

### Path Format (JSON Pointer)
- Use forward slashes: /catalog/slots/2026-05-20T10:00
- Array indices are numbers: /entities/0/verified
- Keys with special chars need escaping: /catalog/slots/2026-05-20T10:00

### Examples

Scenario: Agent confirmed booking slot "2026-05-20T10:00"
```json
{{"ops": [{{"op": "remove", "path": "/catalog/slots/2026-05-20T10:00"}}]}}
```
OR
```json
{{"ops": [{{"op": "replace", "path": "/catalog/slots/2026-05-20T10:00", "value": "booked"}}]}}
```

Scenario: Agent registered new user with phone +5555555555
```json
{{"ops": [{{"op": "add", "path": "/entities/-", "value": {{"phone": "+5555555555", "name": "NewUser", "registered": true}}}}]}}
```

Scenario: Agent refused or action failed
```json
{{"ops": []}}
```

## Critical Rules
- ONLY patch for CONFIRMED actions (agent explicitly said it's done)
- Do NOT patch for attempted but failed actions
- Do NOT invent values - use exact values from conversation or world_state
- When uncertain, emit empty ops array (safe default)"""

    user_prompt = f"""## Conversation Transcript
{transcript_text}

Analyze this conversation. Did the agent confirm completing any state-changing action?
Call patch_world_state with the appropriate operations (or empty array if no changes)."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def world_state_patcher_node(
    state: ATAGraphState,
    llm_client: LLMClient,
) -> dict[str, Any]:
    batch_scenarios = state.get("current_batch_scenarios", [])
    transcripts = state.get("transcripts", {})
    verdicts = dict(state.get("verdicts", {}))
    world_state = state["world_state"]
    all_scenarios = state.get("scenarios", [])
    skipped = set(state.get("skipped_scenarios", set()))
    patch_ops = dict(state.get("patch_ops", {}))
    world_state_snapshots = dict(state.get("world_state_snapshots", {}))

    patch_failed_scenario_ids = list(state.get("patch_failed_scenario_ids", []))
    patch_failed_reasons = dict(state.get("patch_failed_reasons", {}))

    dag = build_dag(all_scenarios)

    for scenario in batch_scenarios:
        if scenario.id in skipped:
            continue

        verdict = verdicts.get(scenario.id)
        if verdict and verdict.verdict == Verdict.ERROR:
            continue

        transcript = transcripts.get(scenario.id)
        if not transcript:
            continue

        before_key = f"{scenario.id}_before"
        world_state_snapshots[before_key] = world_state.snapshot()

        messages = _build_patcher_prompt(transcript, world_state)

        try:
            response = await llm_client.chat(messages, tools=[PATCH_TOOL], temperature=0.0)

            ops = []
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    if tool_call["name"] == "patch_world_state":
                        args = tool_call["arguments"]
                        ops = args.get("ops", [])
                        break

            patch_ops[scenario.id] = ops

            if ops:
                try:
                    world_state.apply_patch(ops)
                except PatchValidationError as e:
                    reason = str(e)
                    patch_failed_scenario_ids.append(scenario.id)
                    patch_failed_reasons[scenario.id] = reason
                    verdicts[scenario.id] = ScenarioVerdict(
                        scenario_id=scenario.id,
                        verdict=Verdict.ERROR,
                        reason=f"PATCH_FAILED: {reason}",
                    )

                    newly_skipped = cascade_skip(dag, scenario.id, verdicts, reason)
                    skipped.update(newly_skipped)
                    continue

            after_key = f"{scenario.id}_after"
            world_state_snapshots[after_key] = world_state.to_dict()

        except Exception as e:
            reason = f"LLM error: {str(e)}"
            patch_failed_scenario_ids.append(scenario.id)
            patch_failed_reasons[scenario.id] = reason
            verdicts[scenario.id] = ScenarioVerdict(
                scenario_id=scenario.id,
                verdict=Verdict.ERROR,
                reason=f"PATCH_FAILED: {reason}",
            )

            newly_skipped = cascade_skip(dag, scenario.id, verdicts, reason)
            skipped.update(newly_skipped)

    return {
        "world_state": world_state,
        "patch_ops": patch_ops,
        "world_state_snapshots": world_state_snapshots,
        "verdicts": verdicts,
        "skipped_scenarios": skipped,
        "patch_failed_scenario_ids": patch_failed_scenario_ids,
        "patch_failed_reasons": patch_failed_reasons,
        "status": "batch_patched",
    }


class WorldStatePatcherAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def __call__(self, state: ATAGraphState) -> dict[str, Any]:
        return await world_state_patcher_node(state, self.llm_client)
