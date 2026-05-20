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
        f"User: {turn.user_message}\nAgent: {turn.agent_response}"
        for turn in transcript.turns
    )

    system_prompt = """You are evaluating whether a conversation transcript satisfies a specific check.
Return a JSON object with:
- satisfied: boolean indicating if the check is satisfied
- reasoning: brief explanation of your evaluation"""

    user_prompt = f"""Transcript:
{transcript_text}

Check to evaluate (for {assertion.speaker}): {assertion.check}

Does this transcript satisfy the check? Focus on the {assertion.speaker}'s messages."""

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
        f"User: {turn.user_message}\nAgent: {turn.agent_response}"
        for turn in transcript.turns
    )

    system_prompt = """You are classifying the agent's behavior in a conversation.
Possible behaviors:
- refusal: Agent declined to perform the requested action
- confirmation: Agent confirmed completing the requested action
- clarification: Agent asked for more information
- escalation: Agent escalated to a human or different system

Return a JSON object with:
- satisfied: boolean indicating if the agent exhibited the expected behavior
- reasoning: brief explanation including which behavior was observed"""

    user_prompt = f"""Transcript:
{transcript_text}

Expected behavior: {assertion.expected_behavior}

Did the agent exhibit this behavior?"""

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

        verdicts[scenario.id] = ScenarioVerdict(
            scenario_id=scenario.id,
            verdict=verdict,
            reason=f"Run {'succeeded' if run_success else 'failed'}, "
                   f"{sum(1 for r in results if r.satisfied)}/{len(results)} assertions satisfied",
            assertion_results=assertion_results,
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
