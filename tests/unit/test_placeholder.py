import pytest

from agentspect.services.placeholder import (
    PlaceholderResolutionError,
    extract_placeholders,
    resolve_placeholders,
    validate_placeholders,
)


def test_resolve_simple_placeholder(sample_world_state):
    text = "Hello, my phone is {{entities/0/phone}}"
    result = resolve_placeholders(text, sample_world_state)
    assert result == "Hello, my phone is +33612345678"


def test_resolve_multiple_placeholders(sample_world_state):
    text = "Name: {{entities/0/name}}, Lang: {{context/language}}"
    result = resolve_placeholders(text, sample_world_state)
    assert result == "Name: Isabelle Martin, Lang: fr"


def test_resolve_placeholder_with_dots(sample_world_state):
    text = "Time: {{context.current_time}}"
    result = resolve_placeholders(text, sample_world_state)
    assert result == "Time: 2026-05-16T08:00"


def test_resolve_placeholder_missing_path(sample_world_state):
    text = "Value: {{nonexistent/path}}"
    with pytest.raises(PlaceholderResolutionError, match="path.*not found"):
        resolve_placeholders(text, sample_world_state)


def test_resolve_no_placeholders(sample_world_state):
    text = "Hello, world!"
    result = resolve_placeholders(text, sample_world_state)
    assert result == "Hello, world!"


def test_resolve_nested_array_index(sample_world_state):
    text = "Slot: {{catalog/available_slots/0}}"
    result = resolve_placeholders(text, sample_world_state)
    assert result == "Slot: 2026-05-20T10:00"


def test_extract_placeholders():
    text = "Name: {{entities/0/name}}, Phone: {{entities/0/phone}}"
    placeholders = extract_placeholders(text)
    assert placeholders == ["entities/0/name", "entities/0/phone"]


def test_extract_no_placeholders():
    text = "Hello, world!"
    placeholders = extract_placeholders(text)
    assert placeholders == []


def test_validate_placeholders_valid(sample_world_state):
    text = "Name: {{entities/0/name}}"
    errors = validate_placeholders(text, sample_world_state)
    assert errors == []


def test_validate_placeholders_invalid(sample_world_state):
    text = "Value: {{nonexistent/path}}"
    errors = validate_placeholders(text, sample_world_state)
    assert len(errors) == 1
    assert "Invalid placeholder path" in errors[0]


def test_resolve_placeholder_with_spaces(sample_world_state):
    text = "Phone: {{ entities/0/phone }}"
    result = resolve_placeholders(text, sample_world_state)
    assert result == "Phone: +33612345678"


def test_resolve_boolean_value(sample_world_state):
    text = "Verified: {{entities/0/verified}}"
    result = resolve_placeholders(text, sample_world_state)
    assert result == "Verified: True"
