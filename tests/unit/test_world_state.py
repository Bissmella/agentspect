import pytest

from agentspect.models.world_state import PatchValidationError, WorldState


def test_snapshot(sample_world_state):
    ws = WorldState(sample_world_state)
    snap = ws.snapshot()
    assert snap == sample_world_state
    assert len(ws.snapshots) == 1

    snap["entities"][0]["name"] = "Modified"
    assert ws.data["entities"][0]["name"] == "Isabelle Martin"


def test_resolve_pointer(sample_world_state):
    ws = WorldState(sample_world_state)
    assert ws.resolve_pointer("/entities/0/phone") == "+33612345678"
    assert ws.resolve_pointer("/catalog/available_slots/0") == "2026-05-20T10:00"
    assert ws.resolve_pointer("/context/language") == "fr"


def test_resolve_pointer_without_leading_slash(sample_world_state):
    ws = WorldState(sample_world_state)
    assert ws.resolve_pointer("entities/0/name") == "Isabelle Martin"


def test_pointer_exists(sample_world_state):
    ws = WorldState(sample_world_state)
    assert ws.pointer_exists("/entities/0") is True
    assert ws.pointer_exists("/nonexistent") is False
    assert ws.pointer_exists("/entities/99") is False


def test_apply_patch_add(sample_world_state):
    ws = WorldState(sample_world_state)
    ops = [{"op": "add", "path": "/catalog/available_slots/-", "value": "2026-05-21T10:00"}]
    ws.apply_patch(ops)
    assert len(ws.data["catalog"]["available_slots"]) == 3
    assert ws.data["catalog"]["available_slots"][-1] == "2026-05-21T10:00"


def test_apply_patch_remove(sample_world_state):
    ws = WorldState(sample_world_state)
    ops = [{"op": "remove", "path": "/catalog/available_slots/0"}]
    ws.apply_patch(ops)
    assert len(ws.data["catalog"]["available_slots"]) == 1
    assert ws.data["catalog"]["available_slots"][0] == "2026-05-20T14:00"


def test_apply_patch_replace(sample_world_state):
    ws = WorldState(sample_world_state)
    ops = [{"op": "replace", "path": "/entities/0/verified", "value": False}]
    ws.apply_patch(ops)
    assert ws.data["entities"][0]["verified"] is False


def test_validate_patch_invalid_path(sample_world_state):
    ws = WorldState(sample_world_state)
    ops = [{"op": "remove", "path": "/nonexistent/path"}]
    errors = ws.validate_patch(ops)
    assert len(errors) == 1
    assert "does not exist" in errors[0]


def test_apply_patch_invalid_fails(sample_world_state):
    ws = WorldState(sample_world_state)
    ops = [{"op": "remove", "path": "/nonexistent/path"}]
    with pytest.raises(PatchValidationError):
        ws.apply_patch(ops)


def test_validate_patch_type_mismatch(sample_world_state):
    ws = WorldState(sample_world_state)
    ops = [{"op": "replace", "path": "/entities/0/verified", "value": "not_a_bool"}]
    errors = ws.validate_patch(ops)
    assert len(errors) == 1
    assert "Type mismatch" in errors[0]


def test_to_dict_returns_copy(sample_world_state):
    ws = WorldState(sample_world_state)
    exported = ws.to_dict()
    exported["entities"][0]["name"] = "Changed"
    assert ws.data["entities"][0]["name"] == "Isabelle Martin"


def test_from_dict(sample_world_state):
    ws = WorldState.from_dict(sample_world_state)
    sample_world_state["entities"][0]["name"] = "Changed"
    assert ws.data["entities"][0]["name"] == "Isabelle Martin"
