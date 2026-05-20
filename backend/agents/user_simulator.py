import asyncio
from typing import Any

from backend.adapters.base import ProtocolAdapter
from backend.agents.state import ATAGraphState
from backend.llm.client import LLMClient
from backend.models.suite import Scenario, ScenarioVerdict, Verdict
from backend.models.transcript import Transcript
from backend.models.world_state import WorldState
from backend.services.placeholder import PlaceholderResolutionError, resolve_placeholders


def _build_persona_prompt(scenario: Scenario, world_state: WorldState) -> str:
    persona_info = ""
    if scenario.persona:
        entities = world_state.data.get("entities", [])
        for entity in entities:
            if entity.get("id") == scenario.persona:
                persona_info = f"\nYou are playing the role of: {entity}"
                break

    context = world_state.data.get("context", {})
    context_str = f"\nContext: {context}" if context else ""

    return f"""You are simulating a user in a conversation with an AI agent.

Scenario: {scenario.description}
Type: {scenario.type.value}
{persona_info}
{context_str}

Your job is to:
1. Stay in character as the persona throughout the conversation
2. Follow the planned turns but adapt naturally if the agent goes off-script
3. Be realistic - respond as a real user would
4. Do not evaluate or judge the agent's responses - just converse

If the agent asks clarifying questions, respond appropriately while staying on task.
If the agent refuses or redirects, acknowledge it naturally."""


async def _run_single_scenario(
    scenario: Scenario,
    world_state: WorldState,
    adapter: ProtocolAdapter,
    llm_client: LLMClient,
) -> tuple[Transcript, ScenarioVerdict | None]:
    try:
        session_id = await adapter.start_session(scenario.id)
    except ConnectionError as e:
        return (
            Transcript(
                scenario_id=scenario.id,
                session_id="",
                protocol="unknown",
                turns=[],
            ),
            ScenarioVerdict(
                scenario_id=scenario.id,
                verdict=Verdict.ERROR,
                reason=f"Connection failed: {str(e)}",
            ),
        )

    transcript = adapter.create_transcript(
        scenario_id=scenario.id,
        session_id=session_id,
        protocol=adapter.__class__.__name__.replace("Adapter", "").lower(),
    )

    world_state_data = world_state.to_dict()
    verdict = None
    conversation_history: list[dict[str, str]] = []

    try:
        for i, turn_template in enumerate(scenario.turns):
            try:
                user_message = resolve_placeholders(turn_template, world_state_data)
            except PlaceholderResolutionError as e:
                verdict = ScenarioVerdict(
                    scenario_id=scenario.id,
                    verdict=Verdict.ERROR,
                    reason=str(e),
                )
                break

            if i > 0 and conversation_history:
                system_prompt = _build_persona_prompt(scenario, world_state)
                adapt_messages = [
                    {"role": "system", "content": system_prompt},
                    *conversation_history,
                    {
                        "role": "user",
                        "content": f"The agent just said: {conversation_history[-1]['content'] if conversation_history else ''}\n\n"
                        f"Your planned message was: {user_message}\n\n"
                        "Adapt your response if needed based on the conversation flow. "
                        "Return only the message you want to send (no explanation).",
                    },
                ]
                response = await llm_client.chat(adapt_messages, temperature=0.3)
                user_message = response.content.strip() or user_message

            turn = await adapter.send_turn(session_id, user_message)
            transcript.add_turn(turn)

            if turn.error:
                verdict = ScenarioVerdict(
                    scenario_id=scenario.id,
                    verdict=Verdict.ERROR,
                    reason=turn.error,
                )
                break

            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "assistant", "content": turn.agent_response})

    finally:
        await adapter.end_session(session_id)
        transcript.finalize()

    return transcript, verdict


async def user_simulator_node(
    state: ATAGraphState,
    llm_client: LLMClient,
    adapter: ProtocolAdapter,
) -> dict[str, Any]:
    batch_scenarios = state.get("current_batch_scenarios", [])
    world_state = state["world_state"]
    skipped = state.get("skipped_scenarios", set())

    transcripts = dict(state.get("transcripts", {}))
    verdicts = dict(state.get("verdicts", {}))

    tasks = []
    scenario_ids = []

    for scenario in batch_scenarios:
        if scenario.id in skipped:
            continue
        tasks.append(
            _run_single_scenario(scenario, world_state, adapter, llm_client)
        )
        scenario_ids.append(scenario.id)

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for scenario_id, result in zip(scenario_ids, results):
            if isinstance(result, Exception):
                verdicts[scenario_id] = ScenarioVerdict(
                    scenario_id=scenario_id,
                    verdict=Verdict.ERROR,
                    reason=f"Execution exception: {str(result)}",
                )
                transcripts[scenario_id] = Transcript(
                    scenario_id=scenario_id,
                    session_id="",
                    protocol="unknown",
                    turns=[],
                )
            else:
                transcript, verdict = result
                transcripts[scenario_id] = transcript
                if verdict:
                    verdicts[scenario_id] = verdict

    return {
        "transcripts": transcripts,
        "verdicts": verdicts,
        "status": "batch_executed",
    }


class UserSimulatorAgent:
    def __init__(self, llm_client: LLMClient, adapter: ProtocolAdapter):
        self.llm_client = llm_client
        self.adapter = adapter

    async def __call__(self, state: ATAGraphState) -> dict[str, Any]:
        return await user_simulator_node(state, self.llm_client, self.adapter)
