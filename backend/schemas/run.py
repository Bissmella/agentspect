import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CreateRunRequest(BaseModel):
    yaml_content: str = Field(description="Raw YAML configuration string")


class CreateRunResponse(BaseModel):
    suite_id: uuid.UUID
    status: str = "pending"


class ScenarioSummary(BaseModel):
    id: uuid.UUID
    type: str
    description: str
    status: str
    verdict: str | None = None
    verdict_reason: str | None = None


class SuiteStatusResponse(BaseModel):
    id: uuid.UUID
    status: str
    created_at: datetime
    agent_name: str
    agent_url: str
    protocol: str
    llm_provider: str
    llm_model: str
    total_scenarios: int
    positive_count: int
    negative_count: int
    completed_at: datetime | None = None
    error_message: str | None = None
    scenarios: list[ScenarioSummary] = Field(default_factory=list)


class ScenarioDetailResponse(BaseModel):
    id: uuid.UUID
    type: str
    description: str
    status: str
    verdict: str | None = None
    verdict_reason: str | None = None
    assertions_json: Any = None
    depends_on: uuid.UUID | None = None
    depends_on_type: str | None = None
    execution_order: int = 0
    patch_ops_json: Any = None


class TurnResponse(BaseModel):
    user: str
    agent: str
    latency_ms: int = 0


class TranscriptResponse(BaseModel):
    scenario_id: uuid.UUID
    turns: list[TurnResponse]


class ReportResponse(BaseModel):
    suite_id: uuid.UUID
    created_at: datetime
    summary: dict[str, Any]
    full_report: dict[str, Any] | None = None
