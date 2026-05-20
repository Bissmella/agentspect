from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from backend.agents.state import ATAGraphState
from backend.llm.client import LLMClient
from backend.models.suite import DependsOnType, ScenarioType, Verdict


class FailureAnalysis(BaseModel):
    summary: str = Field(description="Brief summary of failure patterns")
    constraint_violations: list[str] = Field(
        default_factory=list,
        description="Constraints that were violated",
    )
    entity_lookup_failures: list[str] = Field(
        default_factory=list,
        description="Entity lookups that failed",
    )
    catalog_misuse: list[str] = Field(
        default_factory=list,
        description="Catalog values that were misused",
    )
    recommendations: list[str] = Field(
        default_factory=list,
        description="Recommendations for fixing issues",
    )


def _build_analysis_prompt(state: ATAGraphState) -> list[dict[str, str]]:
    verdicts = state.get("verdicts", {})
    transcripts = state.get("transcripts", {})
    scenarios = state.get("scenarios", [])

    failed_scenarios = []
    for scenario in scenarios:
        verdict = verdicts.get(scenario.id)
        if verdict and verdict.verdict in (
            Verdict.FAILURE,
            Verdict.FAILURE_CORRUPT,
            Verdict.SUSPECT,
        ):
            transcript = transcripts.get(scenario.id)
            transcript_text = ""
            if transcript:
                transcript_text = "\n".join(
                    f"User: {t.user_message}\nAgent: {t.agent_response}"
                    for t in transcript.turns
                )

            failed_scenarios.append({
                "id": scenario.id,
                "type": scenario.type.value,
                "description": scenario.description,
                "verdict": verdict.verdict.value,
                "reason": verdict.reason,
                "transcript": transcript_text,
            })

    if not failed_scenarios:
        return []

    world_state = state.get("world_state_input")
    constraints = world_state.constraints if world_state else []

    system_prompt = """You are analyzing test failures for an AI agent.
Identify patterns in the failures and provide actionable insights.

Focus on:
1. Which constraints were violated
2. Which entity lookups failed (wrong ID, missing data)
3. Which catalog values were misused (invalid options, out-of-range)
4. Recommendations for the agent developer"""

    user_prompt = f"""Constraints defined:
{constraints}

Failed scenarios:
{failed_scenarios}

Analyze these failures and provide structured feedback."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def reporter_node(
    state: ATAGraphState,
    llm_client: LLMClient,
) -> dict[str, Any]:
    verdicts = state.get("verdicts", {})
    transcripts = state.get("transcripts", {})
    scenarios = state.get("scenarios", [])
    world_state_snapshots = state.get("world_state_snapshots", {})
    patch_ops = state.get("patch_ops", {})
    agent_under_test = state.get("agent_under_test")

    verdict_counts = Counter(v.verdict.value for v in verdicts.values())

    scenarios_map = {s.id: s for s in scenarios}
    per_scenario_results = []

    for scenario in scenarios:
        verdict = verdicts.get(scenario.id)
        transcript = transcripts.get(scenario.id)

        result = {
            "id": scenario.id,
            "type": scenario.type.value,
            "description": scenario.description,
            "verdict": verdict.verdict.value if verdict else "pending",
            "reason": verdict.reason if verdict else None,
            "assertion_results": verdict.assertion_results if verdict else [],
            "turns": [
                {"user": t.user_message, "agent": t.agent_response, "latency_ms": t.latency_ms}
                for t in (transcript.turns if transcript else [])
            ],
            "world_state_before": world_state_snapshots.get(f"{scenario.id}_before"),
            "world_state_after": world_state_snapshots.get(f"{scenario.id}_after"),
            "patch_ops": patch_ops.get(scenario.id, []),
        }

        if scenario.depends_on:
            result["depends_on"] = scenario.depends_on
            result["depends_on_type"] = scenario.depends_on_type.value if scenario.depends_on_type else None

        per_scenario_results.append(result)

    probe_chains = []
    for scenario in scenarios:
        if scenario.depends_on and scenario.depends_on_type:
            parent = scenarios_map.get(scenario.depends_on)
            parent_verdict = verdicts.get(scenario.depends_on)
            probe_verdict = verdicts.get(scenario.id)

            probe_chains.append({
                "parent_id": scenario.depends_on,
                "probe_id": scenario.id,
                "probe_type": scenario.depends_on_type.value,
                "parent_verdict": parent_verdict.verdict.value if parent_verdict else "pending",
                "probe_verdict": probe_verdict.verdict.value if probe_verdict else "pending",
            })

    world_state_audit = []
    processed = set()
    for scenario in scenarios:
        if scenario.id in processed:
            continue
        before = world_state_snapshots.get(f"{scenario.id}_before")
        after = world_state_snapshots.get(f"{scenario.id}_after")
        if before or after:
            world_state_audit.append({
                "scenario_id": scenario.id,
                "before": before,
                "after": after,
                "patch_ops": patch_ops.get(scenario.id, []),
            })
        processed.add(scenario.id)

    failure_analysis = None
    analysis_messages = _build_analysis_prompt(state)
    if analysis_messages:
        try:
            analysis = await llm_client.chat_with_structured_output(
                analysis_messages, FailureAnalysis, temperature=0.3
            )
            failure_analysis = analysis.model_dump()
        except Exception:
            failure_analysis = {
                "summary": "Failed to generate failure analysis",
                "constraint_violations": [],
                "entity_lookup_failures": [],
                "catalog_misuse": [],
                "recommendations": [],
            }

    report = {
        "agent_name": agent_under_test.name if agent_under_test else "Unknown",
        "agent_url": agent_under_test.url if agent_under_test else None,
        "verdict_counts": dict(verdict_counts),
        "total_scenarios": len(scenarios),
        "scenarios": per_scenario_results,
        "probe_chains": probe_chains,
        "world_state_audit": world_state_audit,
        "failure_analysis": failure_analysis,
    }

    return {
        "report": report,
        "status": "completed",
    }


class ReporterAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def __call__(self, state: ATAGraphState) -> dict[str, Any]:
        return await reporter_node(state, self.llm_client)
