from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, model_validator


class Verdict(str, Enum):
    SUCCESS = "success"
    SUCCESS_UNVERIFIED = "success_unverified"
    FAILURE = "failure"
    FAILURE_CORRUPT = "failure_corrupt"
    SUSPECT = "suspect"
    ERROR = "error"


class ScenarioType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class DependsOnType(str, Enum):
    PROBE = "probe"
    DEFENSIVE_PROBE = "defensive_probe"


class WorldStateAssertion(BaseModel):
    type: Literal["world_state"] = "world_state"
    description: str
    path: str
    operator: str = Field(pattern=r"^(removed|added|equals|contains|not_contains)$")
    expected_value: Any | None = None


class TranscriptAssertion(BaseModel):
    type: Literal["transcript"] = "transcript"
    description: str
    check: str
    speaker: str = "agent"


class BehavioralAssertion(BaseModel):
    type: Literal["behavioral"] = "behavioral"
    description: str
    expected_behavior: str = Field(
        pattern=r"^(refusal|confirmation|clarification|escalation)$"
    )


Assertion = Annotated[
    Union[WorldStateAssertion, TranscriptAssertion, BehavioralAssertion],
    Field(discriminator="type"),
]


class VoiceActionType(str, Enum):
    BARGE_IN = "barge_in"  # start speaking while the agent is still talking
    DTMF = "dtmf"          # send keypad tones instead of / with speech
    SILENCE = "silence"    # send nothing and wait, to test the agent's reprompt/recovery


class VoiceAction(BaseModel):
    """A timed voice behaviour the tester performs on one turn.

    Attached to a user turn by index and honoured by the duplex voice runtime; the
    lockstep text path ignores it. Additive, so scenarios without voice actions are
    unaffected. Names are transport-neutral.
    """

    type: VoiceActionType
    turn_index: int = Field(ge=0)  # the user turn this action applies to
    at_ms: int | None = Field(default=None, ge=0)  # barge_in: ms into the agent's speech
    dtmf: str | None = None  # dtmf: the digit string to send

    @model_validator(mode="after")
    def _check_required_fields(self):
        if self.type == VoiceActionType.BARGE_IN and self.at_ms is None:
            raise ValueError("barge_in voice action requires 'at_ms'")
        if self.type == VoiceActionType.DTMF and not self.dtmf:
            raise ValueError("dtmf voice action requires 'dtmf' digits")
        return self


class Scenario(BaseModel):
    id: str
    type: ScenarioType
    description: str
    turns: list[str]
    persona: str | None = None
    assertions: list[Assertion] = Field(default_factory=list)
    depends_on: str | None = None
    depends_on_type: DependsOnType | None = None
    target_constraint: str | None = None
    voice_actions: list[VoiceAction] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_voice_action_indices(self):
        for action in self.voice_actions:
            if action.turn_index >= len(self.turns):
                raise ValueError(
                    f"voice_action turn_index {action.turn_index} is out of range "
                    f"for a scenario with {len(self.turns)} turn(s)"
                )
        return self


class ScenarioVerdict(BaseModel):
    scenario_id: str
    verdict: Verdict
    reason: str
    assertion_results: list[dict[str, Any]] = Field(default_factory=list)
    recovery_quality: str | None = None
