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


class DataField(BaseModel):
    """One piece of data the agent was asked to collect, with its ground truth.

    ``expected`` is the value the test *decides* the caller will provide — invented
    by the ScenarioGenerator for diversity and difficulty (unusual spellings,
    dotted/plus-tagged emails, international phone formats), NOT pulled from
    ``world_state``. The generator plants this value in the collection turns so the
    user simulator speaks it, then the recall probe checks the agent captured it.
    """

    name: str  # e.g. "email", "phone", "full_name"
    expected: str  # ground-truth value the caller provides (generator-envisaged)
    kind: str = Field(default="text", pattern=r"^(phone|email|name|text)$")


class DataCollectionAssertion(BaseModel):
    """Checks how accurately the agent captured data the user provided.

    Optional — only scenarios that actually test data collection carry one; agents
    that don't collect data get none. Evaluated on a **recall** transcript (a
    follow-up turn/probe where the agent reads the data back or is asked for it):
    each field's recalled value is extracted and compared to ground truth,
    normalized per ``kind``, with a character/digit error rate (so "one digit off"
    scores as a near-miss, not a binary fail).
    """

    type: Literal["data_collection"] = "data_collection"
    description: str
    fields: list[DataField]
    max_cer: float = Field(default=0.0, ge=0.0, le=1.0)  # per-field tolerance (0 = exact)


Assertion = Annotated[
    Union[
        WorldStateAssertion,
        TranscriptAssertion,
        BehavioralAssertion,
        DataCollectionAssertion,
    ],
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
    data_collection: list[dict[str, Any]] = Field(default_factory=list)
