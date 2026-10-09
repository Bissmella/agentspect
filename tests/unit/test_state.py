import pytest

from agentspect.agents.state import ATAGraphState
from agentspect.models.suite import Scenario, ScenarioType, ScenarioVerdict, Verdict
from agentspect.models.transcript import Transcript
from agentspect.models.world_state import WorldState
from agentspect.models.yaml_input import AgentUnderTest, LLMConfig, TestConfig, WorldStateInput


def test_state_can_be_created_empty():
    state = ATAGraphState()
    assert isinstance(state, dict)


def test_state_accepts_all_fields():
    state = ATAGraphState(
        agent_under_test=AgentUnderTest(
            name="Test",
            url="http://localhost",
            protocol="http",
            description="Test agent",
        ),
        world_state_input=WorldStateInput(
            entities=[{"id": "user_1"}],
            catalog={"items": []},
            constraints=["rule 1"],
            context={"lang": "en"},
        ),
        test_config=TestConfig(total=2, positive=1, negative=1),
        llm_config=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"),
        world_state=WorldState({"entities": []}),
        scenarios=[
            Scenario(
                id="s1",
                type=ScenarioType.POSITIVE,
                description="Test",
                turns=["Hello"],
            )
        ],
        scenario_batches=[["s1"]],
        current_batch_index=0,
        current_batch_scenarios=[],
        transcripts={},
        verdicts={},
        patch_ops={},
        world_state_snapshots={},
        skipped_scenarios=set(),
        patch_failed_scenario_ids=[],
        patch_failed_reasons={},
        report={},
        error=None,
        status="initialized",
    )

    assert state["agent_under_test"].name == "Test"
    assert state["test_config"].total == 2
    assert state["status"] == "initialized"


def test_state_partial_update():
    state = ATAGraphState(status="starting")

    update = {"status": "running", "error": None}
    state.update(update)

    assert state["status"] == "running"
    assert state.get("error") is None


def test_state_with_verdicts():
    state = ATAGraphState(
        verdicts={
            "s1": ScenarioVerdict(
                scenario_id="s1",
                verdict=Verdict.SUCCESS,
                reason="All passed",
            ),
            "s2": ScenarioVerdict(
                scenario_id="s2",
                verdict=Verdict.FAILURE,
                reason="Assertion failed",
            ),
        }
    )

    assert state["verdicts"]["s1"].verdict == Verdict.SUCCESS
    assert state["verdicts"]["s2"].verdict == Verdict.FAILURE


def test_state_with_transcripts():
    state = ATAGraphState(
        transcripts={
            "s1": Transcript(
                scenario_id="s1",
                session_id="session-123",
                protocol="http",
                turns=[],
            )
        }
    )

    assert state["transcripts"]["s1"].scenario_id == "s1"
    assert state["transcripts"]["s1"].protocol == "http"
