import asyncio
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.config import settings
from backend.db.models import Report, Scenario, Suite, SuiteStatus
from backend.dependencies import close_redis, get_db
from backend.pubsub import get_event_history, subscribe_events
from backend.schemas.run import (
    CreateRunRequest,
    CreateRunResponse,
    ReportResponse,
    ScenarioDetailResponse,
    ScenarioSummary,
    SuiteStatusResponse,
    TranscriptResponse,
    TurnResponse,
)
from backend.services.blob import async_download_json, create_s3_client, ensure_bucket
from agentspect import YAMLValidationError, parse_and_validate
from backend.tasks import run_suite_task

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    s3 = create_s3_client()
    await asyncio.to_thread(ensure_bucket, s3, settings.s3_bucket_name)
    yield
    await close_redis()


app = FastAPI(title="ATA - Agent Testing Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/runs", response_model=CreateRunResponse)
async def create_run(
    request: CreateRunRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        yaml_input, yaml_hash = parse_and_validate(request.yaml_content)
    except YAMLValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    agent = yaml_input.agent_under_test
    test_cfg = yaml_input.test_config
    llm_cfg = yaml_input.llm_config

    suite = Suite(
        status=SuiteStatus.PENDING,
        yaml_input_hash=yaml_hash,
        agent_name=agent.name,
        agent_url=agent.url,
        protocol=agent.protocol,
        llm_provider=llm_cfg.provider,
        llm_model=llm_cfg.model,
        total_scenarios=test_cfg.total,
        positive_count=test_cfg.positive,
        negative_count=test_cfg.negative,
    )
    db.add(suite)
    await db.commit()
    await db.refresh(suite)

    task = run_suite_task.delay(str(suite.id), request.yaml_content)
    suite.celery_task_id = task.id
    await db.commit()

    return CreateRunResponse(suite_id=suite.id)


@app.get("/api/runs/{run_id}", response_model=SuiteStatusResponse)
async def get_run(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Suite)
        .where(Suite.id == run_id)
        .options(selectinload(Suite.scenarios))
    )
    result = await db.execute(stmt)
    suite = result.scalar_one_or_none()
    if not suite:
        raise HTTPException(status_code=404, detail="Suite not found")

    scenarios = [
        ScenarioSummary(
            id=sc.id,
            type=sc.type.value,
            description=sc.description,
            status=sc.status.value,
            verdict=sc.verdict.value if sc.verdict else None,
            verdict_reason=sc.verdict_reason,
        )
        for sc in suite.scenarios
    ]

    return SuiteStatusResponse(
        id=suite.id,
        status=suite.status.value,
        created_at=suite.created_at,
        agent_name=suite.agent_name,
        agent_url=suite.agent_url,
        protocol=suite.protocol.value if hasattr(suite.protocol, "value") else suite.protocol,
        llm_provider=suite.llm_provider,
        llm_model=suite.llm_model,
        total_scenarios=suite.total_scenarios,
        positive_count=suite.positive_count,
        negative_count=suite.negative_count,
        completed_at=suite.completed_at,
        error_message=suite.error_message,
        scenarios=scenarios,
    )


@app.get("/api/runs/{run_id}/report", response_model=ReportResponse)
async def get_report(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Suite).where(Suite.id == run_id)
    result = await db.execute(stmt)
    suite = result.scalar_one_or_none()
    if not suite:
        raise HTTPException(status_code=404, detail="Suite not found")

    stmt = select(Report).where(Report.suite_id == run_id)
    result = await db.execute(stmt)
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not available yet")

    full_report = None
    if report.full_report_blob_key:
        try:
            full_report = await async_download_json(report.full_report_blob_key)
        except Exception:
            logger.warning("Failed to fetch full report from S3 for suite %s", run_id)

    return ReportResponse(
        suite_id=run_id,
        created_at=report.created_at,
        summary=report.summary_json,
        full_report=full_report,
    )


@app.get("/api/runs/{run_id}/scenarios", response_model=list[ScenarioDetailResponse])
async def get_scenarios(
    run_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Suite).where(Suite.id == run_id)
    result = await db.execute(stmt)
    suite = result.scalar_one_or_none()
    if not suite:
        raise HTTPException(status_code=404, detail="Suite not found")

    stmt = select(Scenario).where(Scenario.suite_id == run_id).order_by(Scenario.execution_order)
    result = await db.execute(stmt)
    scenarios = result.scalars().all()

    return [
        ScenarioDetailResponse(
            id=sc.id,
            type=sc.type.value,
            description=sc.description,
            status=sc.status.value,
            verdict=sc.verdict.value if sc.verdict else None,
            verdict_reason=sc.verdict_reason,
            assertions_json=sc.assertions_json,
            depends_on=sc.depends_on,
            depends_on_type=sc.depends_on_type.value if sc.depends_on_type else None,
            execution_order=sc.execution_order,
            patch_ops_json=sc.patch_ops_json,
        )
        for sc in scenarios
    ]


@app.get(
    "/api/runs/{run_id}/scenarios/{scenario_id}/transcript",
    response_model=TranscriptResponse,
)
async def get_transcript(
    run_id: uuid.UUID,
    scenario_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Scenario).where(
        Scenario.id == scenario_id,
        Scenario.suite_id == run_id,
    )
    result = await db.execute(stmt)
    scenario = result.scalar_one_or_none()
    if not scenario:
        raise HTTPException(status_code=404, detail="Scenario not found")

    if not scenario.transcript_blob_key:
        raise HTTPException(status_code=404, detail="Transcript not available")

    try:
        turns_data = await async_download_json(scenario.transcript_blob_key)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to retrieve transcript from storage")

    turns = [
        TurnResponse(
            user=t.get("user", ""),
            agent=t.get("agent", ""),
            latency_ms=t.get("latency_ms", 0),
        )
        for t in turns_data
    ]

    return TranscriptResponse(scenario_id=scenario_id, turns=turns)


@app.websocket("/api/runs/{run_id}/ws")
async def run_progress_ws(
    websocket: WebSocket,
    run_id: uuid.UUID,
):
    await websocket.accept()

    try:
        history = await get_event_history(str(run_id))
        for event in history:
            await websocket.send_json(event)

        terminal_events = {"suite_completed", "suite_failed"}
        if history and history[-1].get("event") in terminal_events:
            await websocket.close()
            return

        seen_seqs = {e.get("seq") for e in history if "seq" in e}

        async for event in subscribe_events(str(run_id)):
            seq = event.get("seq")
            if seq and seq in seen_seqs:
                continue
            seen_seqs.add(seq)

            await websocket.send_json(event)

            if event.get("event") in terminal_events:
                await websocket.close()
                return

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocket error for suite %s", run_id)
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
