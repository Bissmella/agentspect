import asyncio
from typing import Any

from ata.adapters.base import ProtocolAdapter
from ata.agents.state import ATAGraphState
from ata.llm.client import LLMClient
from ata.models.suite import Scenario, ScenarioVerdict, Verdict
from ata.models.transcript import AgentFailure, Transcript
from ata.models.world_state import WorldState
from ata.services.placeholder import PlaceholderResolutionError, resolve_placeholders


def _build_persona_prompt(scenario: Scenario, world_state: WorldState) -> str:
    import json

    persona_info = ""
    persona_data = None
    if scenario.persona:
        entities = world_state.data.get("entities", [])
        for entity in entities:
            if entity.get("id") == scenario.persona:
                persona_data = entity
                persona_info = f"""
## Your Character
You are: {entity.get('name', 'Unknown')}
Your details: {json.dumps(entity, indent=2)}

CRITICAL: When asked for your phone number, name, or any personal info, use EXACTLY the values above.
- Your phone: {entity.get('phone', 'not specified')}
- Your name: {entity.get('name', 'not specified')}
Do NOT make up different values."""
                break

    context = world_state.data.get("context", {})
    context_str = ""
    if context:
        context_str = f"\n## Context\nLanguage: {context.get('language', 'en')}\nTimezone: {context.get('timezone', 'UTC')}"

    scenario_type_guidance = ""
    if scenario.type.value == "positive":
        scenario_type_guidance = """
## Expected Outcome
This is a POSITIVE test - the agent SHOULD successfully complete your request.
Cooperate with the agent and provide accurate information when asked."""
    else:
        scenario_type_guidance = """
## Expected Outcome
This is a NEGATIVE test - the agent SHOULD refuse or fail your request.
You may be using invalid data intentionally. Stay in character and don't correct yourself."""

    return f"""You are simulating a real user talking to an AI assistant.

## Scenario
{scenario.description}
{persona_info}
{context_str}
{scenario_type_guidance}

## Your Behavior
1. Act like a real person, not a test robot
2. Use your EXACT persona details when asked (phone, name, etc.)
3. Respond naturally to questions - don't just repeat the script
4. If the agent asks for clarification, provide it using your persona data
5. Stay on task - you're trying to accomplish the scenario goal
6. Keep responses concise (1-3 sentences)

## What NOT to Do
- Do NOT make up fake phone numbers or names
- Do NOT reveal you are a test or simulation
- Do NOT evaluate or critique the agent's responses"""


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

            # `message` is what we hand the adapter: a plain string, or — for a
            # streaming (voice) adapter on an adapted turn — the LLM token stream
            # itself, so tokens flow straight into streaming TTS.
            message: str | Any = user_message
            streaming = getattr(adapter, "supports_token_stream", False)
            if i > 0 and conversation_history:
                system_prompt = _build_persona_prompt(scenario, world_state)
                last_agent_response = conversation_history[-1]['content'] if conversation_history else ''
                adapt_messages = [
                    {"role": "system", "content": system_prompt},
                    *conversation_history,
                    {
                        "role": "user",
                        "content": f"""The agent just responded: "{last_agent_response}"

Your next planned message was: "{user_message}"

Adapt your response naturally based on what the agent said.
- If the agent asked a question, answer it using your EXACT persona details
- If the agent confirmed something, acknowledge it
- If the agent refused, respond naturally (don't argue unless that's your goal)
- Stay on task toward your scenario goal

IMPORTANT: Use your real persona data (phone, name, etc.) - do not make up values.

Return ONLY the message you want to send. No explanations or quotes.""",
                    },
                ]
                if streaming:
                    message = llm_client.chat_stream(adapt_messages, temperature=0.3)
                else:
                    response = await llm_client.chat(adapt_messages, temperature=0.3)
                    user_message = response.content.strip() or user_message
                    message = user_message

            is_last_turn = i == len(scenario.turns) - 1
            turn = await adapter.send_turn(session_id, message)
            # For a streamed turn, recover what was actually said from the adapter.
            if not isinstance(message, str):
                user_message = turn.user_message or user_message

            
            blank = not turn.agent_response.strip()
            if turn.error is None and turn.agent_failure is None and blank and not is_last_turn:
                turn.agent_failure = AgentFailure.NO_RESPONSE

            if turn.agent_failure == AgentFailure.DISCONNECTED and is_last_turn:
                turn.agent_failure = None

            transcript.add_turn(turn)

            # Framework/transport fault on ATA's side: cannot complete the run.
            if turn.error:
                verdict = ScenarioVerdict(
                    scenario_id=scenario.id,
                    verdict=Verdict.ERROR,
                    reason=turn.error,
                )
                break

            # Agent behavioral failure: stop the conversation but let the scorer
            # score it as a failed run rather than discarding it as an ERROR.
            if turn.agent_failure is not None:
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
