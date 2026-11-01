"""EntityPopulatorAgent — generate concrete entities into world_state from context.

Runs once, before scenario generation, when the suite declares ``entity_specs``.
This node turns each spec into realistic instances and
APPENDS them to world_state.entities. Entities authored concretely are preserved,
never replaced.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from agentspect.agents.state import ATAGraphState
from agentspect.llm.client import LLMClient
from agentspect.models.world_state import WorldState
from agentspect.models.yaml_input import EntitySpec, TestConfig

logger = logging.getLogger(__name__)


class GeneratedEntities(BaseModel):
    entities: list[dict[str, Any]] = Field(default_factory=list)


def _default_count(test_config: TestConfig) -> int:
    # A small shared pool, not one entity per scenario: constraint, capability, and
    # out-of-scope tests reuse or need no entity, so sizing to positive count overshoots.
    return max(2, min(test_config.positive, 5))


def _ensure_unique_ids(entities: list[dict[str, Any]], kind: str, taken: set) -> list[dict[str, Any]]:
    out = []
    n = 0
    for e in entities:
        if not isinstance(e, dict):
            continue
        e = dict(e)
        eid = e.get("id")
        while not eid or eid in taken:
            n += 1
            eid = f"{kind}_{n}"
        e["id"] = eid
        taken.add(eid)
        out.append(e)
    return out


async def _generate_for_spec(
    llm_client: LLMClient,
    spec: EntitySpec,
    count: int,
    context: dict[str, Any],
    constraints: list[str],
) -> list[dict[str, Any]]:
    language = context.get("language")
    country = context.get("country")
    locale = ", ".join(
        b for b in (f"language {language}" if language else "", f"country {country}" if country else "") if b
    )
    locale_rule = (
        f"- Locale is {locale}. Names, phone number formats, addresses, honorifics, and every "
        "locale-sensitive value MUST match it.\n"
        if locale
        else ""
    )
    persona_rule = (
        "- Each entity is a person: add a short `persona` field (name, demeanor, speaking style) "
        "written in the context's language.\n"
        if spec.kind == "person"
        else ""
    )
    system_prompt = f"""You generate realistic world_state entities for black-box testing \
of a conversational agent. The designer gave a shape and the runtime context, not concrete \
values. The entities must belong in that context.

Runtime context (authoritative — generate data that fits this world):
{context}

Rules:
- Produce exactly {count} entities of kind "{spec.kind}".
- Every entity MUST have a stable, unique `id`.
- Give each entity these attributes: {', '.join(spec.attributes) if spec.attributes else '(infer sensible ones)'}.
{locale_rule}- Values must be realistic and consistent with the context above. Never use placeholder \
values like "John Doe" or "+1111111111".
- Span the constraints so both valid and boundary cases are possible (e.g. some verified \
and some unverified), rather than {count} identical entities.
{persona_rule}"""

    user_prompt = f"""Constraints: {constraints}
Extra guidance: {spec.traits or "(none)"}

Generate the entities now, consistent with the runtime context above."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        result = await llm_client.chat_with_structured_output(
            messages, GeneratedEntities, temperature=0.7
        )
        return result.entities
    except Exception:
        # A failed generation must not kill the run — the suite proceeds with whatever
        # concrete entities already exist. ponytail: no deterministic faker fallback yet.
        logger.warning("Entity generation failed for spec kind=%s", spec.kind, exc_info=True)
        return []


async def entity_populator_node(
    state: ATAGraphState,
    llm_client: LLMClient,
) -> dict[str, Any]:
    specs: list[EntitySpec] = state.get("entity_specs", [])
    world_state: WorldState = state["world_state"]

    if not specs:
        return {"world_state": world_state, "status": "no_entity_specs"}

    data = world_state.data
    existing = data.setdefault("entities", [])
    context = data.get("context", {})
    constraints = data.get("constraints", [])
    test_config = state["test_config"]

    taken = {e.get("id") for e in existing if isinstance(e, dict)}
    generated: list[dict[str, Any]] = []
    for spec in specs:
        count = spec.count or _default_count(test_config)
        raw = await _generate_for_spec(llm_client, spec, count, context, constraints)
        generated.extend(_ensure_unique_ids(raw, spec.kind, taken))

    existing.extend(generated)  # augment, never replace

    return {"world_state": world_state, "status": "entities_populated"}


class EntityPopulatorAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def __call__(self, state: ATAGraphState) -> dict[str, Any]:
        return await entity_populator_node(state, self.llm_client)
