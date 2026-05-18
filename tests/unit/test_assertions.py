import pytest
from pydantic import ValidationError

from backend.models.suite import (
    Assertion,
    BehavioralAssertion,
    Scenario,
    ScenarioType,
    TranscriptAssertion,
    WorldStateAssertion,
)


def test_world_state_assertion_valid():
    assertion = WorldStateAssertion(
        description="Slot should be removed",
        path="/catalog/available_slots/0",
        operator="removed",
    )
    assert assertion.type == "world_state"
    assert assertion.operator == "removed"


def test_world_state_assertion_with_expected_value():
    assertion = WorldStateAssertion(
        description="Booking count should equal 1",
        path="/entities/0/bookings",
        operator="equals",
        expected_value=1,
    )
    assert assertion.expected_value == 1


def test_world_state_assertion_invalid_operator():
    with pytest.raises(ValidationError):
        WorldStateAssertion(
            description="Test",
            path="/test",
            operator="invalid_op",
        )


def test_transcript_assertion_valid():
    assertion = TranscriptAssertion(
        description="Agent mentions confirmation",
        check="agent mentioned a confirmation number",
    )
    assert assertion.type == "transcript"
    assert assertion.speaker == "agent"


def test_transcript_assertion_user_speaker():
    assertion = TranscriptAssertion(
        description="User confirms",
        check="user said yes",
        speaker="user",
    )
    assert assertion.speaker == "user"


def test_behavioral_assertion_valid():
    assertion = BehavioralAssertion(
        description="Agent refuses invalid request",
        expected_behavior="refusal",
    )
    assert assertion.type == "behavioral"


def test_behavioral_assertion_invalid_behavior():
    with pytest.raises(ValidationError):
        BehavioralAssertion(
            description="Test",
            expected_behavior="invalid_behavior",
        )


def test_discriminated_union_world_state():
    data = {
        "type": "world_state",
        "description": "Test",
        "path": "/test",
        "operator": "removed",
    }
    from pydantic import TypeAdapter
    adapter = TypeAdapter(Assertion)
    assertion = adapter.validate_python(data)
    assert isinstance(assertion, WorldStateAssertion)


def test_discriminated_union_transcript():
    data = {
        "type": "transcript",
        "description": "Test",
        "check": "agent said something",
    }
    from pydantic import TypeAdapter
    adapter = TypeAdapter(Assertion)
    assertion = adapter.validate_python(data)
    assert isinstance(assertion, TranscriptAssertion)


def test_discriminated_union_behavioral():
    data = {
        "type": "behavioral",
        "description": "Test",
        "expected_behavior": "confirmation",
    }
    from pydantic import TypeAdapter
    adapter = TypeAdapter(Assertion)
    assertion = adapter.validate_python(data)
    assert isinstance(assertion, BehavioralAssertion)


def test_scenario_with_mixed_assertions():
    scenario = Scenario(
        id="test_scenario",
        type=ScenarioType.POSITIVE,
        description="Test scenario with multiple assertion types",
        turns=["Hello", "Book a slot"],
        assertions=[
            WorldStateAssertion(
                description="Slot removed",
                path="/catalog/available_slots/0",
                operator="removed",
            ),
            TranscriptAssertion(
                description="Agent confirms",
                check="agent provided confirmation number",
            ),
            BehavioralAssertion(
                description="Agent confirms booking",
                expected_behavior="confirmation",
            ),
        ],
    )
    assert len(scenario.assertions) == 3
    assert isinstance(scenario.assertions[0], WorldStateAssertion)
    assert isinstance(scenario.assertions[1], TranscriptAssertion)
    assert isinstance(scenario.assertions[2], BehavioralAssertion)


def test_scenario_serialization():
    scenario = Scenario(
        id="test",
        type=ScenarioType.NEGATIVE,
        description="Test",
        turns=["Hello"],
        assertions=[
            WorldStateAssertion(
                description="Test",
                path="/test",
                operator="equals",
                expected_value="foo",
            ),
        ],
    )
    data = scenario.model_dump()
    assert data["assertions"][0]["type"] == "world_state"
    assert data["assertions"][0]["expected_value"] == "foo"

    reconstructed = Scenario.model_validate(data)
    assert isinstance(reconstructed.assertions[0], WorldStateAssertion)
