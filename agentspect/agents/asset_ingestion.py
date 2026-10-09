"""AssetIngestionAgent — turn bounded asset samples into world_state material.

Runs once, before scenario generation, when the suite declares assets. It does
not read whole files: the deterministic loader already produced a small sample.
This node transforms that sample per the asset's role and merges it into
world_state:

- ``entities``     — an LLM maps sample rows into world_state entities.
- ``data_sample``  — the sampled rows are placed as-is (deterministic).
- ``knowledge_base`` — reserved for RAG; raises a clear NotImplementedError.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentspect.agents.state import ATAGraphState
from agentspect.assets.base import Asset
from agentspect.llm.client import LLMClient
from agentspect.models.world_state import WorldState


class IngestedEntities(BaseModel):
    entities: list[dict[str, Any]] = Field(default_factory=list)


def _place_at_pointer(data: dict[str, Any], pointer: str, value: list) -> None:
    parts = [p for p in pointer.strip("/").split("/") if p != ""]
    if not parts:
        return
    cur = data
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    last = parts[-1]
    existing = cur.get(last)
    if isinstance(existing, list):
        existing.extend(value)
    else:
        cur[last] = value


def _ensure_ids(rows: list[dict[str, Any]], asset_id: str) -> list[dict[str, Any]]:
    out = []
    for i, row in enumerate(rows):
        row = dict(row)
        if "id" not in row:
            row["id"] = f"{asset_id}_{i}"
        out.append(row)
    return out


async def _ingest_entities(
    asset: Asset,
    llm_client: LLMClient,
) -> list[dict[str, Any]]:
    if not asset.rows:
        return []

    system_prompt = """You convert sample rows from an external dataset into world_state \
entities for black-box testing of a conversational agent.

Rules:
- Produce one entity per useful sample row (skip rows that are clearly unusable).
- Every entity MUST have a stable, unique `id`.
- Keep the real values from the rows — never invent data.
- Include the attributes an agent would look up or verify.
- Preserve variety across the rows so both valid and boundary/negative cases are possible."""

    user_prompt = f"""Dataset description: {asset.description or "(none provided)"}

Sample rows ({len(asset.rows)} of {asset.total_rows} total):
{asset.rows}

Return the entities."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        result = await llm_client.chat_with_structured_output(
            messages, IngestedEntities, temperature=0.0
        )
        entities = result.entities
    except Exception:
        entities = []

    if not entities:
        # Deterministic fallback: use the raw sampled rows as entities.
        entities = asset.rows

    return _ensure_ids(entities, asset.id)


async def asset_ingestion_node(
    state: ATAGraphState,
    llm_client: LLMClient,
) -> dict[str, Any]:
    assets: list[Asset] = state.get("assets", [])
    world_state: WorldState = state["world_state"]

    if not assets:
        return {"status": "no_assets"}

    data = world_state.data

    for asset in assets:
        if asset.role == "entities":
            entities = await _ingest_entities(asset, llm_client)
            target = asset.target or "/entities"
            _place_at_pointer(data, target, entities)

        elif asset.role == "data_sample":
            target = asset.target or f"/catalog/{asset.id}"
            _place_at_pointer(data, target, asset.rows)

        elif asset.role == "knowledge_base":
            raise NotImplementedError(
                f"Asset '{asset.id}': role 'knowledge_base' (RAG) is not implemented yet"
            )

    return {"world_state": world_state, "status": "assets_ingested"}


class AssetIngestionAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def __call__(self, state: ATAGraphState) -> dict[str, Any]:
        return await asset_ingestion_node(state, self.llm_client)
