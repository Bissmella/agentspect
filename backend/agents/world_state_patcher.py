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
    transcript_text = "\n".join(
        f"User: {turn.user_message}\nAgent: {turn.agent_response}"
        for turn in transcript.turns
    )

    world_state_json = world_state.to_dict()

    system_prompt = f"""You are analyzing a conversation transcript to infer what changes should be made to the world state.

Current world_state:
{world_state_json}

Your task:
1. Analyze the conversation to determine if any actions were completed
2. If the agent confirmed completing an action (booking, update, deletion, etc.), emit the corresponding patch operations
3. If the agent refused or the action was not completed, emit NO operations
4. Only emit operations for COMPLETED actions, not attempted ones

Rules for patches:
- Use JSON Pointer paths (e.g., /entities/0/verified, /catalog/available_slots/0)
- For removing items from arrays, use "remove" with the index path
- For updating values, use "replace"
- For adding new items, use "add"
- Paths must exist in the current world_state (for remove/replace/move)
- Do not invent values - only use values from the conversation or world_state

If uncertain about any mutation, emit an empty ops array. Freezing state is safer than a wrong mutation."""

    user_prompt = f"""Conversation transcript:
{transcript_text}

Analyze this conversation and call the patch_world_state tool with the appropriate operations.
If no state changes should occur, call it with an empty ops array."""

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

    patch_failed = False
    patch_failed_scenario_id = None
    patch_failed_reason = None

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
                    patch_failed = True
                    patch_failed_scenario_id = scenario.id
                    patch_failed_reason = str(e)

                    dag = build_dag(all_scenarios)
                    newly_skipped = cascade_skip(dag, scenario.id, verdicts, str(e))
                    skipped.update(newly_skipped)

                    break

            after_key = f"{scenario.id}_after"
            world_state_snapshots[after_key] = world_state.to_dict()

        except Exception as e:
            patch_failed = True
            patch_failed_scenario_id = scenario.id
            patch_failed_reason = f"LLM error: {str(e)}"

            dag = build_dag(all_scenarios)
            newly_skipped = cascade_skip(dag, scenario.id, verdicts, str(e))
            skipped.update(newly_skipped)
            break

    return {
        "world_state": world_state,
        "patch_ops": patch_ops,
        "world_state_snapshots": world_state_snapshots,
        "verdicts": verdicts,
        "skipped_scenarios": skipped,
        "patch_failed": patch_failed,
        "patch_failed_scenario_id": patch_failed_scenario_id,
        "patch_failed_reason": patch_failed_reason,
        "status": "batch_patched",
    }


class WorldStatePatcherAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def __call__(self, state: ATAGraphState) -> dict[str, Any]:
        return await world_state_patcher_node(state, self.llm_client)
