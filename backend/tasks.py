import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from backend.celery_app import celery
from backend.config import settings
from backend.db.models import (
    DependsOnTypeDB,
    Report,
    Scenario,
    ScenarioStatus,
    ScenarioTypeDB,
    Suite,
    SuiteStatus,
    VerdictDB,
)
from backend.pubsub import publish_event
from backend.services.blob import create_s3_client, ensure_bucket, upload_json

logger = logging.getLogger(__name__)

_task_engine = create_async_engine(settings.database_url, poolclass=NullPool)
_task_session = async_sessionmaker(_task_engine, class_=AsyncSession, expire_on_commit=False)


def _scenario_uuid(suite_id: uuid.UUID, scenario_str_id: str) -> uuid.UUID:
    return uuid.uuid5(suite_id, scenario_str_id)


def _map_verdict(verdict_str: str) -> VerdictDB | None:
    mapping = {
        "success": VerdictDB.SUCCESS,
        "success_unverified": VerdictDB.SUCCESS_UNVERIFIED,
        "failure": VerdictDB.FAILURE,
        "failure_corrupt": VerdictDB.FAILURE_CORRUPT,
        "suspect": VerdictDB.SUSPECT,
        "error": VerdictDB.ERROR,
    }
    return mapping.get(verdict_str)


def _map_scenario_type(type_str: str) -> ScenarioTypeDB:
    return ScenarioTypeDB.POSITIVE if type_str == "positive" else ScenarioTypeDB.NEGATIVE


def _map_depends_on_type(type_str: str | None) -> DependsOnTypeDB | None:
    if type_str == "probe":
        return DependsOnTypeDB.PROBE
    if type_str == "defensive_probe":
        return DependsOnTypeDB.DEFENSIVE_PROBE
    return None


def _make_progress_callback(suite_id: str):
    def callback(event: str, data: dict[str, Any]) -> None:
        node = data.get("node")
        status = data.get("status")

        if event == "starting_execution":
            publish_event(suite_id, "suite_started")
        elif event.startswith("node_generate_scenarios"):
            publish_event(suite_id, "scenarios_generated", {"status": status})
        elif event == "node_prepare_batch":
            publish_event(suite_id, "batch_started", {"status": status})
        elif event == "node_user_simulator":
            publish_event(suite_id, "batch_executed", {"status": status})
        elif event == "node_scorer":
            publish_event(suite_id, "batch_scored", {"status": status})
        elif event == "node_world_state_patcher":
            if data.get("error"):
                publish_event(suite_id, "patch_failed", {"error": data["error"]})
            else:
                publish_event(suite_id, "patch_applied", {"status": status})
        elif event == "node_advance_batch":
            publish_event(suite_id, "batch_completed", {"status": status})
        elif event == "node_reporter":
            publish_event(suite_id, "report_generating")
        elif event == "completed":
            pass
        elif event in ("parsing_yaml", "yaml_validated", "initialized"):
            publish_event(suite_id, event, data)

    return callback


async def _store_results(
    suite_id: uuid.UUID,
    report: dict[str, Any],
) -> None:
    s3 = create_s3_client()
    ensure_bucket(s3, settings.s3_bucket_name)

    sid = str(suite_id)
    scenario_results = report.get("scenarios", [])

    blob_keys: dict[str, dict[str, str | None]] = {}

    for sc in scenario_results:
        sc_str_id = sc["id"]
        sc_uuid = _scenario_uuid(suite_id, sc_str_id)
        sc_uuid_str = str(sc_uuid)
        keys: dict[str, str | None] = {}

        if sc.get("turns"):
            key = f"suites/{sid}/transcripts/{sc_uuid_str}.json"
            upload_json(s3, key, sc["turns"])
            keys["transcript"] = key

        if sc.get("world_state_before"):
            key = f"suites/{sid}/world_state/{sc_uuid_str}_before.json"
            upload_json(s3, key, sc["world_state_before"])
            keys["ws_before"] = key

        if sc.get("world_state_after"):
            key = f"suites/{sid}/world_state/{sc_uuid_str}_after.json"
            upload_json(s3, key, sc["world_state_after"])
            keys["ws_after"] = key

        blob_keys[sc_str_id] = keys

    report_key = f"suites/{sid}/report.json"
    upload_json(s3, report_key, report)

    async with _task_session() as session:
        id_map: dict[str, uuid.UUID] = {}
        for idx, sc in enumerate(scenario_results):
            sc_uuid = _scenario_uuid(suite_id, sc["id"])
            id_map[sc["id"]] = sc_uuid

        for idx, sc in enumerate(scenario_results):
            sc_uuid = id_map[sc["id"]]
            keys = blob_keys.get(sc["id"], {})

            depends_on_uuid = None
            if sc.get("depends_on"):
                depends_on_uuid = id_map.get(sc["depends_on"])

            scenario_row = Scenario(
                id=sc_uuid,
                suite_id=suite_id,
                type=_map_scenario_type(sc["type"]),
                description=sc["description"],
                persona_entity_id=sc.get("persona"),
                turns_json=sc.get("turns", []),
                assertions_json=sc.get("assertion_results", []),
                depends_on=depends_on_uuid,
                depends_on_type=_map_depends_on_type(sc.get("depends_on_type")),
                execution_order=idx,
                status=ScenarioStatus.COMPLETED if sc["verdict"] != "error" else ScenarioStatus.SKIPPED,
                verdict=_map_verdict(sc["verdict"]),
                verdict_reason=sc.get("reason"),
                transcript_blob_key=keys.get("transcript"),
                world_state_before_key=keys.get("ws_before"),
                world_state_after_key=keys.get("ws_after"),
                patch_ops_json=sc.get("patch_ops"),
            )
            session.add(scenario_row)

        summary_json = {
            "verdict_counts": report.get("verdict_counts", {}),
            "total_scenarios": report.get("total_scenarios", 0),
            "agent_name": report.get("agent_name"),
            "probe_chains": report.get("probe_chains", []),
            "failure_analysis": report.get("failure_analysis"),
        }

        report_row = Report(
            suite_id=suite_id,
            summary_json=summary_json,
            full_report_blob_key=report_key,
        )
        session.add(report_row)

        suite = await session.get(Suite, suite_id)
        if suite:
            suite.status = SuiteStatus.COMPLETED
            suite.completed_at = datetime.now(timezone.utc)

        await session.commit()


async def _mark_suite_failed(suite_id: uuid.UUID, error_message: str) -> None:
    async with _task_session() as session:
        suite = await session.get(Suite, suite_id)
        if suite:
            suite.status = SuiteStatus.FAILED
            suite.error_message = error_message
            suite.completed_at = datetime.now(timezone.utc)
            await session.commit()


async def _execute_suite(suite_id_str: str, yaml_str: str) -> dict[str, Any]:
    suite_id = uuid.UUID(suite_id_str)

    async with _task_session() as session:
        suite = await session.get(Suite, suite_id)
        if suite:
            suite.status = SuiteStatus.RUNNING
            await session.commit()

    publish_event(suite_id_str, "suite_running")

    from backend.agents.orchestrator import OrchestratorAgent

    progress_callback = _make_progress_callback(suite_id_str)
    orchestrator = OrchestratorAgent(yaml_str, progress_callback)

    try:
        report = await orchestrator.run()
    except Exception as e:
        logger.exception("Suite %s failed", suite_id_str)
        await _mark_suite_failed(suite_id, str(e))
        publish_event(suite_id_str, "suite_failed", {"error": str(e)})
        raise

    await _store_results(suite_id, report)
    publish_event(suite_id_str, "suite_completed", {
        "verdict_counts": report.get("verdict_counts", {}),
    })

    return report


@celery.task(bind=True, name="backend.tasks.run_suite_task")
def run_suite_task(self, suite_id: str, yaml_str: str) -> dict:
    return asyncio.run(_execute_suite(suite_id, yaml_str))
