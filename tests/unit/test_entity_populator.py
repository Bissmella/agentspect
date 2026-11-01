import pytest
from unittest.mock import AsyncMock, MagicMock

from agentspect.agents.entity_populator import GeneratedEntities, entity_populator_node
from agentspect.agents.state import ATAGraphState
from agentspect.models.world_state import WorldState
from agentspect.models.yaml_input import EntitySpec, TestConfig


def _state(entities, specs) -> ATAGraphState:
    return ATAGraphState(
        world_state=WorldState.from_dict({
            "entities": list(entities),
            "catalog": {},
            "constraints": ["only verified callers can book"],
            "context": {"language": "fr", "country": "FR"},
        }),
        entity_specs=specs,
        test_config=TestConfig(total=4, positive=3, negative=1),
    )


@pytest.fixture
def mock_llm():
    c = MagicMock()
    c.chat_with_structured_output = AsyncMock()
    return c


@pytest.mark.asyncio
async def test_appends_and_preserves_concrete_entities(mock_llm):
    mock_llm.chat_with_structured_output.return_value = GeneratedEntities(entities=[
        {"id": "gen_a", "name": "Isabelle Martin", "phone": "+33612345678", "verified": True},
        {"id": "gen_b", "name": "Jean Dupont", "phone": "+33698765432", "verified": False},
    ])
    pinned = {"id": "pinned_1", "name": "Pinned Caller", "phone": "+33600000000"}
    state = _state([pinned], [EntitySpec(kind="person", count=2, attributes=["name", "phone"])])

    result = await entity_populator_node(state, mock_llm)

    entities = result["world_state"].data["entities"]
    ids = [e["id"] for e in entities]
    assert result["status"] == "entities_populated"
    assert entities[0] == pinned, "concrete entity preserved and untouched, not replaced"
    assert "gen_a" in ids and "gen_b" in ids


@pytest.mark.asyncio
async def test_context_reaches_the_model(mock_llm):
    mock_llm.chat_with_structured_output.return_value = GeneratedEntities(entities=[{"id": "g1"}])
    state = _state([], [EntitySpec(count=1)])  # context is language fr, country FR
    await entity_populator_node(state, mock_llm)
    messages = mock_llm.chat_with_structured_output.call_args.args[0]
    blob = " ".join(m["content"] for m in messages)
    assert "fr" in blob and "FR" in blob, "language and country from world_state.context drive generation"


@pytest.mark.asyncio
async def test_skips_when_no_specs(mock_llm):
    pinned = {"id": "pinned_1", "name": "X"}
    state = _state([pinned], [])
    result = await entity_populator_node(state, mock_llm)
    assert result["status"] == "no_entity_specs"
    assert result["world_state"].data["entities"] == [pinned]
    mock_llm.chat_with_structured_output.assert_not_called()


@pytest.mark.asyncio
async def test_dedupes_and_fills_missing_ids(mock_llm):
    mock_llm.chat_with_structured_output.return_value = GeneratedEntities(entities=[
        {"name": "No Id Given"},
        {"id": "pinned_1", "name": "Clashing Id"},
    ])
    pinned = {"id": "pinned_1", "name": "Pinned"}
    state = _state([pinned], [EntitySpec(count=2)])
    result = await entity_populator_node(state, mock_llm)
    ids = [e["id"] for e in result["world_state"].data["entities"]]
    assert len(ids) == len(set(ids)), "all ids unique"
    assert ids.count("pinned_1") == 1, "original kept, the clashing generated id was reassigned"


@pytest.mark.asyncio
async def test_survives_llm_failure(mock_llm):
    mock_llm.chat_with_structured_output.side_effect = RuntimeError("provider down")
    pinned = {"id": "pinned_1"}
    state = _state([pinned], [EntitySpec(count=2)])
    result = await entity_populator_node(state, mock_llm)
    assert result["world_state"].data["entities"] == [pinned], "concrete entities survive a failed generation"
    assert result["status"] == "entities_populated"
