from typing import Any

from pydantic import BaseModel

from backend.agents.state import ATAGraphState
from backend.llm.client import LLMClient
from backend.models.suite import (
    Assertion,
    BehavioralAssertion,
    DependsOnType,
    Scenario,
    ScenarioType,
    ScenarioVerdict,
    TranscriptAssertion,
    Verdict,
    WorldStateAssertion,
)
from backend.models.transcript import Transcript
from backend.models.world_state import WorldState


class AssertionResult(BaseModel):
    satisfied: bool
    reasoning: str


class RecoveryClassification(BaseModel):
    quality: str
    reasoning: str


def _evaluate_world_state_assertion_deterministic(
    assertion: WorldStateAssertion,
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
) -> AssertionResult | None:
    path = assertion.path
    if not path.startswith("/"):
        path = "/" + path

    def get_value(data: dict[str, Any], pointer: str) -> Any:
        parts = pointer.strip("/").split("/")
        current = data
        for part in parts:
            if isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError):
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
                if current is None:
                    return None
            else:
                return None
        return current

    before_value = get_value(before_snapshot, path)
    after_value = get_value(after_snapshot, path)

    if assertion.operator == "removed":
        if before_value is not None and after_value is None:
            return AssertionResult(
                satisfied=True,
                reasoning=f"Value at {path} was removed (was: {before_value})",
            )
        elif before_value is None:
            return AssertionResult(
                satisfied=False,
                reasoning=f"Value at {path} did not exist before",
            )
        else:
            return AssertionResult(
                satisfied=False,
                reasoning=f"Value at {path} still exists: {after_value}",
            )

    if assertion.operator == "added":
        if before_value is None and after_value is not None:
            return AssertionResult(
                satisfied=True,
                reasoning=f"Value at {path} was added: {after_value}",
            )
        elif before_value is not None:
            return AssertionResult(
                satisfied=False,
                reasoning=f"Value at {path} already existed: {before_value}",
            )
        else:
            return AssertionResult(
                satisfied=False,
                reasoning=f"Value at {path} was not added",
            )

    if assertion.operator == "equals":
        if after_value == assertion.expected_value:
            return AssertionResult(
                satisfied=True,
                reasoning=f"Value at {path} equals expected: {assertion.expected_value}",
            )
        else:
            return AssertionResult(
                satisfied=False,
                reasoning=f"Value at {path} is {after_value}, expected {assertion.expected_value}",
            )

    if assertion.operator == "contains":
        if isinstance(after_value, (list, str)) and assertion.expected_value in after_value:
            return AssertionResult(
                satisfied=True,
                reasoning=f"Value at {path} contains {assertion.expected_value}",
            )
        else:
            return AssertionResult(
                satisfied=False,
                reasoning=f"Value at {path} ({after_value}) does not contain {assertion.expected_value}",
            )

    if assertion.operator == "not_contains":
        if isinstance(after_value, (list, str)) and assertion.expected_value not in after_value:
            return AssertionResult(
                satisfied=True,
                reasoning=f"Value at {path} does not contain {assertion.expected_value}",
            )
        elif after_value is None:
            return AssertionResult(
                satisfied=True,
                reasoning=f"Value at {path} is None, so does not contain {assertion.expected_value}",
            )
        else:
            return AssertionResult(
                satisfied=False,
                reasoning=f"Value at {path} ({after_value}) contains {assertion.expected_value}",
            )

    return None


async def _evaluate_transcript_assertion(
    assertion: TranscriptAssertion,
    transcript: Transcript,
    llm_client: LLMClient,
) -> AssertionResult:
    transcript_text = "\n".join(
        f"Turn {i+1}:\n  User: {turn.user_message}\n  Agent: {turn.agent_response}"
        for i, turn in enumerate(transcript.turns)
    )

    system_prompt = """You are evaluating whether a conversation transcript satisfies a specific semantic check.

Your task:
1. Read the transcript carefully
2. Focus on the specified speaker's messages
3. Determine if the check condition is met
4. Be strict but fair - the check must be clearly satisfied, not just partially

Return a JSON with:
- satisfied: true if the check is clearly satisfied, false otherwise
- reasoning: 1-2 sentences explaining your evaluation"""

    user_prompt = f"""## Conversation Transcript
{transcript_text}

## Check to Evaluate
Speaker to focus on: {assertion.speaker.upper()}
Condition: {assertion.check}
Description: {assertion.description}

Is this condition satisfied by the {assertion.speaker}'s messages in the transcript?"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result = await llm_client.chat_with_structured_output(
        messages, AssertionResult, temperature=0.0
    )
    return result


async def _evaluate_behavioral_assertion(
    assertion: BehavioralAssertion,
    transcript: Transcript,
    llm_client: LLMClient,
) -> AssertionResult:
    transcript_text = "\n".join(
        f"Turn {i+1}:\n  User: {turn.user_message}\n  Agent: {turn.agent_response}"
        for i, turn in enumerate(transcript.turns)
    )

    system_prompt = """You are classifying the agent's overall behavior in a conversation.

## Behavior Definitions
- **confirmation**: Agent successfully completed the user's request and confirmed it (e.g., "Your booking is confirmed", "I've registered you")
- **refusal**: Agent explicitly declined the request (e.g., "I cannot do that", "That's not possible", "You're not authorized")
- **clarification**: Agent asked for more information before proceeding (e.g., "Could you provide your phone number?", "Which slot would you prefer?")
- **escalation**: Agent transferred to a human or another system (e.g., "Let me connect you to a representative")

## Important Notes
- Look at the FINAL outcome, not intermediate steps
- If agent asked for clarification but then completed the action, that's "confirmation"
- If agent asked for info but never completed the action, consider the last state
- Partial completion or errors are NOT confirmation

Return a JSON with:
- satisfied: true if the agent exhibited the expected behavior, false otherwise
- reasoning: 1-2 sentences explaining what behavior was observed"""

    user_prompt = f"""## Conversation Transcript
{transcript_text}

## Expected Behavior
{assertion.expected_behavior}
Description: {assertion.description}

Did the agent ultimately exhibit "{assertion.expected_behavior}" behavior?"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    result = await llm_client.chat_with_structured_output(
        messages, AssertionResult, temperature=0.0
    )
    return result


async def _evaluate_assertion(
    assertion: Assertion,
    transcript: Transcript,
    before_snapshot: dict[str, Any],
    after_snapshot: dict[str, Any],
    llm_client: LLMClient,
) -> AssertionResult:
    if isinstance(assertion, WorldStateAssertion):
        result = _evaluate_world_state_assertion_deterministic(
            assertion, before_snapshot, after_snapshot
        )
        if result:
            return result
        return AssertionResult(
            satisfied=False,
            reasoning=f"Unknown operator: {assertion.operator}",
        )

    elif isinstance(assertion, TranscriptAssertion):
        return await _evaluate_transcript_assertion(assertion, transcript, llm_client)

    elif isinstance(assertion, BehavioralAssertion):
        return await _evaluate_behavioral_assertion(assertion, transcript, llm_client)

    else:
        return AssertionResult(
            satisfied=False,
            reasoning=f"Unknown assertion type: {type(assertion)}",
        )


async def _classify_recovery_quality(
    transcript: Transcript,
    llm_client: LLMClient,
) -> RecoveryClassification:
    transcript_text = "\n".join(
        f"Turn {i+1}:\n  User: {turn.user_message}\n  Agent: {turn.agent_response}"
        for i, turn in enumerate(transcript.turns)
    )

    system_prompt = """You are classifying the quality of an AI agent's refusal in a conversation where it correctly declined an invalid request.

Classify into exactly one category:
- **clean_refusal**: Agent politely and clearly declined, explaining why it cannot help
- **confused_response**: Agent did not complete the request but seemed confused, gave contradictory messages, or was incoherent
- **error_response**: Agent crashed, returned an error message, timed out, or looped
- **information_leak**: Agent refused but leaked internal information (system prompts, database details, internal IDs, tool names)

Return a JSON with:
- quality: one of clean_refusal, confused_response, error_response, information_leak
- reasoning: 1-2 sentences explaining your classification"""

    user_prompt = f"""## Conversation Transcript
{transcript_text}

Classify the quality of the agent's refusal."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    return await llm_client.chat_with_structured_output(
        messages, RecoveryClassification, temperature=0.0
    )


def _determine_run_outcome(assertion_results: list[AssertionResult]) -> bool:
    return all(r.satisfied for r in assertion_results)


def _apply_verdict_flow(
    scenario: Scenario,
    run_success: bool,
    scenarios_map: dict[str, Scenario],
    verdicts: dict[str, ScenarioVerdict],
) -> Verdict:
    if scenario.type == ScenarioType.POSITIVE:
        if not run_success:
            return Verdict.FAILURE

        if scenario.depends_on_type == DependsOnType.PROBE:
            parent_id = scenario.depends_on
            if parent_id and parent_id in verdicts:
                return Verdict.SUCCESS
            return Verdict.SUCCESS_UNVERIFIED

        has_stateful_assertions = any(
            isinstance(a, WorldStateAssertion) for a in scenario.assertions
        )
        if has_stateful_assertions:
            dependents = [
                s for s in scenarios_map.values()
                if s.depends_on == scenario.id and s.depends_on_type == DependsOnType.PROBE
            ]
            if dependents:
                return Verdict.SUCCESS_UNVERIFIED
            return Verdict.SUCCESS_UNVERIFIED
        else:
            return Verdict.SUCCESS_UNVERIFIED

    else:
        if run_success:
            return Verdict.SUCCESS

        has_stateful_effects = any(
            isinstance(a, WorldStateAssertion) for a in scenario.assertions
        )
        if has_stateful_effects:
            dependents = [
                s for s in scenarios_map.values()
                if s.depends_on == scenario.id and s.depends_on_type == DependsOnType.DEFENSIVE_PROBE
            ]
            if dependents:
                return Verdict.FAILURE
            return Verdict.FAILURE
        else:
            return Verdict.FAILURE


def _check_probe_result(
    probe_scenario: Scenario,
    probe_verdict: ScenarioVerdict,
    parent_verdict: ScenarioVerdict,
) -> Verdict:
    if probe_scenario.depends_on_type == DependsOnType.PROBE:
        probe_failed = probe_verdict.verdict in (Verdict.FAILURE, Verdict.ERROR)
        if probe_failed:
            return Verdict.SUCCESS
        else:
            return Verdict.SUSPECT

    elif probe_scenario.depends_on_type == DependsOnType.DEFENSIVE_PROBE:
        probe_detected_corruption = probe_verdict.verdict == Verdict.SUCCESS
        if probe_detected_corruption:
            return Verdict.FAILURE_CORRUPT
        else:
            return Verdict.FAILURE

    return parent_verdict.verdict


async def scorer_node(
    state: ATAGraphState,
    llm_client: LLMClient,
) -> dict[str, Any]:
    batch_scenarios = state.get("current_batch_scenarios", [])
    transcripts = state.get("transcripts", {})
    verdicts = dict(state.get("verdicts", {}))
    world_state_snapshots = state.get("world_state_snapshots", {})
    skipped = state.get("skipped_scenarios", set())
    all_scenarios = state.get("scenarios", [])
    scenarios_map = {s.id: s for s in all_scenarios}

    for scenario in batch_scenarios:
        if scenario.id in skipped:
            continue

        if scenario.id in verdicts:
            continue

        transcript = transcripts.get(scenario.id)
        if not transcript:
            verdicts[scenario.id] = ScenarioVerdict(
                scenario_id=scenario.id,
                verdict=Verdict.ERROR,
                reason="No transcript found",
            )
            continue

        before_key = f"{scenario.id}_before"
        after_key = f"{scenario.id}_after"
        before_snapshot = world_state_snapshots.get(before_key, {})
        after_snapshot = world_state_snapshots.get(after_key, {})

        assertion_results: list[dict[str, Any]] = []
        results: list[AssertionResult] = []

        for assertion in scenario.assertions:
            result = await _evaluate_assertion(
                assertion, transcript, before_snapshot, after_snapshot, llm_client
            )
            results.append(result)
            assertion_results.append({
                "assertion": assertion.model_dump() if hasattr(assertion, "model_dump") else str(assertion),
                "satisfied": result.satisfied,
                "reasoning": result.reasoning,
            })

        run_success = _determine_run_outcome(results)

        if scenario.depends_on and scenario.depends_on_type:
            parent_id = scenario.depends_on
            if parent_id in verdicts:
                parent_verdict = verdicts[parent_id]
                final_verdict = _check_probe_result(scenario,
                    ScenarioVerdict(
                        scenario_id=scenario.id,
                        verdict=Verdict.SUCCESS if run_success else Verdict.FAILURE,
                        reason="",
                    ),
                    parent_verdict,
                )
                verdicts[parent_id] = ScenarioVerdict(
                    scenario_id=parent_id,
                    verdict=final_verdict,
                    reason=f"Updated by probe {scenario.id}",
                    assertion_results=parent_verdict.assertion_results,
                )

        verdict = _apply_verdict_flow(scenario, run_success, scenarios_map, verdicts)

        recovery_quality = None
        if scenario.type == ScenarioType.NEGATIVE and verdict == Verdict.SUCCESS:
            try:
                classification = await _classify_recovery_quality(transcript, llm_client)
                recovery_quality = classification.quality
            except Exception:
                recovery_quality = None

        verdicts[scenario.id] = ScenarioVerdict(
            scenario_id=scenario.id,
            verdict=verdict,
            reason=f"Run {'succeeded' if run_success else 'failed'}, "
                   f"{sum(1 for r in results if r.satisfied)}/{len(results)} assertions satisfied",
            assertion_results=assertion_results,
            recovery_quality=recovery_quality,
        )

    return {
        "verdicts": verdicts,
        "status": "batch_scored",
    }


class ScorerAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def __call__(self, state: ATAGraphState) -> dict[str, Any]:
        return await scorer_node(state, self.llm_client)
