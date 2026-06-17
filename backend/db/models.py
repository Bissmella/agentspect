import uuid
from datetime import datetime
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SuiteStatus(str, PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ScenarioStatus(str, PyEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class ScenarioTypeDB(str, PyEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class VerdictDB(str, PyEnum):
    SUCCESS = "success"
    SUCCESS_UNVERIFIED = "success_unverified"
    FAILURE = "failure"
    FAILURE_CORRUPT = "failure_corrupt"
    SUSPECT = "suspect"
    ERROR = "error"


class DependsOnTypeDB(str, PyEnum):
    PROBE = "probe"
    DEFENSIVE_PROBE = "defensive_probe"


class ProtocolDB(str, PyEnum):
    HTTP = "http"
    WEBSOCKET = "websocket"


class Suite(Base):
    __tablename__ = "suites"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    status: Mapped[SuiteStatus] = mapped_column(
        Enum(SuiteStatus), default=SuiteStatus.PENDING
    )
    yaml_input_hash: Mapped[str] = mapped_column(String(64))
    agent_name: Mapped[str] = mapped_column(String(255))
    agent_url: Mapped[str] = mapped_column(String(2048))
    protocol: Mapped[ProtocolDB] = mapped_column(Enum(ProtocolDB))
    llm_provider: Mapped[str] = mapped_column(String(50))
    llm_model: Mapped[str] = mapped_column(String(100))
    total_scenarios: Mapped[int] = mapped_column(Integer)
    positive_count: Mapped[int] = mapped_column(Integer)
    negative_count: Mapped[int] = mapped_column(Integer)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    scenarios: Mapped[list["Scenario"]] = relationship(back_populates="suite")
    report: Mapped["Report | None"] = relationship(back_populates="suite")


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    suite_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("suites.id")
    )
    type: Mapped[ScenarioTypeDB] = mapped_column(Enum(ScenarioTypeDB))
    description: Mapped[str] = mapped_column(Text)
    persona_entity_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    turns_json: Mapped[dict] = mapped_column(JSONB)
    assertions_json: Mapped[dict] = mapped_column(JSONB)
    depends_on: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("scenarios.id"), nullable=True
    )
    depends_on_type: Mapped[DependsOnTypeDB | None] = mapped_column(
        Enum(DependsOnTypeDB), nullable=True
    )
    execution_order: Mapped[int] = mapped_column(Integer)
    status: Mapped[ScenarioStatus] = mapped_column(
        Enum(ScenarioStatus), default=ScenarioStatus.PENDING
    )
    verdict: Mapped[VerdictDB | None] = mapped_column(Enum(VerdictDB), nullable=True)
    verdict_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    transcript_blob_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    world_state_before_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    world_state_after_key: Mapped[str | None] = mapped_column(
        String(512), nullable=True
    )
    patch_ops_json: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    target_constraint: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    recovery_quality: Mapped[str | None] = mapped_column(String(64), nullable=True)

    suite: Mapped["Suite"] = relationship(back_populates="scenarios")
    parent_scenario: Mapped["Scenario | None"] = relationship(
        remote_side="Scenario.id", foreign_keys=[depends_on]
    )


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    suite_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("suites.id"), unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    summary_json: Mapped[dict] = mapped_column(JSONB)
    full_report_blob_key: Mapped[str] = mapped_column(String(512))

    suite: Mapped["Suite"] = relationship(back_populates="report")
