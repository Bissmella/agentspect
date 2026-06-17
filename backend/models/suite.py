from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field


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


class ScenarioVerdict(BaseModel):
    scenario_id: str
    verdict: Verdict
    reason: str
    assertion_results: list[dict[str, Any]] = Field(default_factory=list)
    recovery_quality: str | None = None
