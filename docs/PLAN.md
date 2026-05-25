# ATA Implementation Plan

## Context

ATA (Agent Testing Agent) is a greenfield black-box E2E testing framework for conversational agents. Only a `CLAUDE.md` spec exists — no code yet. The spec has 6 identified issues that need fixes before implementation begins. This plan covers: fixing the spec, then building the full system in 7 phases.

---

## Step 0: Fix CLAUDE.md (6 issues)

Update `CLAUDE.md` with these corrections before writing any code:

### Fix 1 — ScorerAgent uses LLM
Move ScorerAgent from the "deterministic" list to the "uses LLM" list. The verdict flow logic (positive/negative routing, probe chains) stays deterministic Python — the LLM is used only to evaluate whether a transcript satisfies each assertion.

### Fix 2 — Define assertion schema
Add three assertion types as a discriminated union:
- **`world_state`** — checks a JSON pointer path changed/exists/equals after the scenario (`path`, `operator`: removed|added|equals|contains|not_contains, optional `expected_value`)
- **`transcript`** — LLM evaluates a semantic `check` string against the transcript (e.g. "agent mentioned confirmation number")
- **`behavioral`** — LLM classifies agent behavior against `expected_behavior`: refusal|confirmation|clarification|escalation

### Fix 3 — PATCH_FAILED cascade
When WorldStatePatcherAgent fails, all transitively dependent scenarios (BFS on DAG) are skipped with verdict `ERROR` and reason `"PATCH_FAILED cascade from {id}"`. The suite continues for unaffected scenarios. World_state stays frozen at last known good state.

### Fix 4 — Defensive probes for negative scenarios
New verdict flow branch: if a negative scenario produces FAILURE (agent incorrectly accepted) AND the scenario has stateful side effects, fire a defensive probe. New verdict `FAILURE_CORRUPT` if probe detects state corruption. New field `depends_on_type`: `"probe"` | `"defensive_probe"`.

### Fix 5 — test_config validation
Validate `total == positive + negative` during YAML parsing. Reject with clear error if mismatch. Also validate `total >= 1`, both counts `>= 0`.

### Fix 6 — Placeholder resolution
`{{path}}` placeholders resolved at execution time (not generation time) against current world_state. Uses JSON pointer. If path doesn't resolve → scenario verdict `ERROR` with "unresolved placeholder" reason.

---

## Phase 1: Foundation — Models, Parsing, Infrastructure (Days 1-3)

**Goal:** Domain models, YAML validation, DAG logic, DB schema, Docker infra. All testable via unit tests.

### Files to create

| File | Purpose |
|------|---------|
| `pyproject.toml` | uv project config with all deps |
| `.python-version` | Pin to 3.12 |
| `.env.example` | Template for env vars (DB URL, API keys, Redis, S3) |
| `docker-compose.yml` | Postgres 16, Redis 7, MinIO (S3), backend, celery worker, frontend |
| `Dockerfile.backend` | Multi-stage: uv install + uvicorn/celery entrypoint |
| `alembic.ini` | Alembic config |
| `backend/__init__.py` | |
| `backend/config.py` | pydantic-settings `BaseSettings` for all env vars |
| `backend/models/__init__.py` | |
| `backend/models/yaml_input.py` | Pydantic models: `AgentUnderTest`, `WorldStateInput`, `TestConfig`, `LLMConfig` |
| `backend/models/suite.py` | `Scenario`, `Assertion` (discriminated union of 3 types), `Verdict` enum (SUCCESS, SUCCESS_UNVERIFIED, FAILURE, FAILURE_CORRUPT, SUSPECT, ERROR), `ScenarioVerdict` |
| `backend/models/world_state.py` | `WorldState` wrapper: `snapshot()`, `apply_patch(ops)`, `resolve_pointer(path)` |
| `backend/models/transcript.py` | `Turn`, `Transcript` |
| `backend/services/__init__.py` | |
| `backend/services/yaml_parser.py` | `parse_and_validate(yaml_str)` — safe_load, Pydantic validation, test_config check (fix #5), rag key rejection |
| `backend/services/placeholder.py` | `resolve_placeholders(text, world_state)` (fix #6) |
| `backend/services/dag.py` | `build_dag(scenarios)`, `topological_batches(dag)`, `cascade_skip(dag, failed_id, verdicts)` (fix #3) |
| `backend/db/__init__.py` | |
| `backend/db/session.py` | Async engine + sessionmaker (asyncpg) |
| `backend/db/models.py` | SQLAlchemy ORM: `suites`, `scenarios`, `reports` tables |
| `backend/db/migrations/env.py` | Alembic async env |

### Database tables

**`suites`**: id (UUID PK), created_at, status (pending/running/completed/failed), yaml_input_hash, agent_name, agent_url, protocol, llm_provider, llm_model, total_scenarios, positive_count, negative_count, celery_task_id, completed_at, error_message

**`scenarios`**: id (UUID PK), suite_id (FK), type (positive/negative), description, persona_entity_id, turns_json (JSONB), assertions_json (JSONB), depends_on (FK self), depends_on_type (probe/defensive_probe), execution_order, status, verdict, verdict_reason, started_at, completed_at, transcript_blob_key, world_state_before_key, world_state_after_key, patch_ops_json (JSONB)

**`reports`**: id (UUID PK), suite_id (FK unique), created_at, summary_json (JSONB), full_report_blob_key

### Dependencies (pyproject.toml)

```
fastapi, uvicorn[standard], sqlalchemy[asyncio], asyncpg, alembic,
celery[redis], redis, pydantic>=2.0, pydantic-settings, pyyaml,
jsonpatch, jsonpointer, langgraph, langchain-core,
anthropic, openai, google-generativeai,
boto3, httpx, websockets
```

Dev: `pytest, pytest-asyncio, pytest-cov, ruff, moto`

### Unit tests

- `tests/unit/test_yaml_parser.py` — valid YAML, total mismatch, rag key, missing keys
- `tests/unit/test_world_state.py` — patch apply, snapshot, failed patch
- `tests/unit/test_dag.py` — topological sort, parallel batches, cascade skip
- `tests/unit/test_placeholder.py` — valid resolution, missing paths
- `tests/unit/test_assertions.py` — serialization of all 3 assertion types

**Verify:** `uv run pytest tests/unit/` passes. `docker compose up postgres redis minio` starts clean.

---

## Phase 2: LLM Client + Protocol Adapters (Days 4-5)

**Goal:** Common LLM interface for 3 providers. Protocol adapters talk to mock agents.

### Files

| File | Purpose |
|------|---------|
| `backend/llm/__init__.py` | |
| `backend/llm/client.py` | `LLMClient` ABC with `chat()` and `chat_with_structured_output()`. Concrete: `AnthropicClient`, `OpenAIClient`, `GoogleClient`. Factory: `create_llm_client(provider, model)`. API keys from env. |
| `backend/adapters/__init__.py` | |
| `backend/adapters/base.py` | `ProtocolAdapter` ABC: `start_session()`, `send_turn(session_id, msg) -> Turn`, `end_session()` |
| `backend/adapters/http_adapter.py` | httpx.AsyncClient, session cookies, configurable timeout |
| `backend/adapters/ws_adapter.py` | websockets lib, one connection per session |
| `tests/e2e/mock_agent_http.py` | Trivial FastAPI app following a script |
| `tests/e2e/mock_agent_ws.py` | Trivial WS server |

### Key interface — LLMClient

```python
class LLMClient(ABC):
    async def chat(self, messages, tools=None, temperature=0.7, max_tokens=4096) -> LLMResponse
    async def chat_with_structured_output(self, messages, output_schema: type[BaseModel], temperature=0.0) -> BaseModel
```

**Verify:** Adapters produce correct Transcript against mock agents. LLM factory routes correctly.

---

## Phase 3: Agent Nodes (Days 6-10)

Each agent is a LangGraph node: takes `ATAGraphState`, returns partial dict update.

### 3a. ScenarioGeneratorAgent (Days 6-7)
`backend/agents/scenario_generator.py`
- Uses `chat_with_structured_output` to produce `list[Scenario]`
- Enforces: correct positive/negative counts, no scenarios for known_limitations, probe generation for positive stateful scenarios, **defensive probe generation for negative scenarios** (fix #4)
- Validates depends_on references and placeholder paths

### 3b. UserSimulatorAgent (Days 7-8)
`backend/agents/user_simulator.py`
- Resolves placeholders (fix #6) at execution time, before each turn
- Runs conversation turn-by-turn via adapter, stays in persona
- Parallel scenarios within a batch via `asyncio.gather`
- Timeout/connection error → Turn.error set → verdict ERROR

### 3c. ScorerAgent (Days 8-9)
`backend/agents/scorer.py` **(fix #1 — LLM-assisted)**
- Per assertion: LLM evaluates `{satisfied: bool, reasoning: str}`
- WorldStateAssertions with simple operators (removed, equals) can short-circuit deterministically
- Run outcome: success if all assertions satisfied, failure otherwise
- Verdict flow is deterministic code applied to the run outcome + scenario type + probe status
- Includes FAILURE_CORRUPT branch for defensive probes (fix #4)

### 3d. WorldStatePatcherAgent (Day 9)
`backend/agents/world_state_patcher.py`
- LLM infers mutations via `patch_world_state` tool call
- Deterministic validation: paths exist, types match, no invented values
- Failure → PATCH_FAILED, do not apply, freeze state
- Uncertain → empty ops (safe default)

### 3e. ReporterAgent (Day 10)
`backend/agents/reporter.py`
- Collects all verdicts, transcripts, patches, snapshots
- LLM synthesizes failure analysis narrative
- Outputs: verdict counts, per-scenario breakdown, world_state audit trail, probe chain outcomes

**Each agent gets unit tests with mocked LLM returning canned responses.**

---

## Phase 4: LangGraph Orchestration (Days 11-13)

**Goal:** Wire all nodes into a StateGraph. Full pipeline works in-process using langgraph and langchain.

### Files

| File | Purpose |
|------|---------|
| `backend/agents/__init__.py` | |
| `backend/agents/state.py` | `ATAGraphState` TypedDict — central contract for all nodes |
| `backend/agents/orchestrator.py` | StateGraph definition, compile, entry point |

### Graph flow

```
START → parse_and_validate → generate_scenarios → execute_batch → score_batch
  → patch_batch → check_cascade → batch_router
      ↳ more batches? → execute_batch (loop)
      ↳ done → generate_report → END
```

### Parallel execution design
- `topological_batches()` groups independent scenarios into parallel batches
- Within a batch: scenarios run concurrently via `asyncio.gather` (world_state is read-only during execution)
- Patch application happens sequentially after the batch completes — no locking needed
- Next batch sees updated world_state

### Integration tests
- Full graph with mocked LLM + mock agent → verify correct verdicts
- PATCH_FAILED cascade → dependents skipped
- Placeholder failure → ERROR verdict

---

## Phase 5: Celery + Redis + FastAPI API (Days 14-16)

### Files

| File | Purpose |
|------|---------|
| `backend/celery_app.py` | Celery instance, Redis broker/backend |
| `backend/tasks.py` | `run_suite` task — instantiates graph, runs it |
| `backend/pubsub.py` | Redis pub/sub helpers: `publish_event()`, `subscribe_events()` |
| `backend/main.py` | FastAPI app with routes |
| `backend/dependencies.py` | Depends: db session, redis, S3 |
| `backend/schemas/__init__.py` | |
| `backend/schemas/run.py` | API request/response Pydantic models |
| `backend/services/blob.py` | S3 upload/download via boto3 |

### API routes

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/runs` | Accept YAML, validate, enqueue Celery task, return suite_id |
| GET | `/api/runs/{id}` | Suite status + scenario verdicts |
| GET | `/api/runs/{id}/report` | Final report |
| GET | `/api/runs/{id}/scenarios` | All scenarios with verdicts |
| GET | `/api/runs/{id}/scenarios/{sid}/transcript` | Transcript from S3 |
| WS | `/api/runs/{id}/ws` | Live progress via Redis pub/sub |

### Progress events (Redis pub/sub channel `ata:suite:{id}`)
`suite_started`, `scenarios_generated`, `batch_started`, `scenario_started`, `scenario_turn`, `scenario_completed`, `patch_applied`, `patch_failed`, `batch_completed`, `report_generating`, `report_ready`, `suite_completed`, `suite_failed`

---

## Phase 6: React Frontend (Days 17-20)

| File | Purpose |
|------|---------|
| `frontend/package.json` | react, react-router-dom, @tanstack/react-query, axios, codemirror, tailwindcss |
| `frontend/vite.config.js` | Proxy `/api` to backend |
| `frontend/src/pages/NewRun.jsx` | CodeMirror YAML editor + provider/model selector → POST /api/runs |
| `frontend/src/pages/RunDashboard.jsx` | WebSocket connection, live scenario progress timeline |
| `frontend/src/pages/Report.jsx` | Verdict summary, per-scenario breakdown, transcript viewer, world_state diffs |
| `Dockerfile.frontend` | Node build + nginx serve |

---

## Phase 7: E2E Testing + Polish (Days 21-23)

- `tests/e2e/test_full_run.py` — POST YAML → WS events → GET report → verify verdicts, S3 blobs, Postgres records
- Test all 6 spec fixes end-to-end
- `examples/booking_agent.yaml`, `examples/faq_agent.yaml`
- `README.md` with setup instructions

---

## Docker Compose services

`postgres:16`, `redis:7-alpine`, `minio/minio` (S3-compatible), `backend` (FastAPI, uvicorn), `celery_worker`, `migrate` (Alembic, profile-gated), `frontend` (Vite build + nginx)

---

## Critical files (in implementation priority order)

1. `backend/agents/state.py` — ATAGraphState TypedDict, central contract for all nodes
2. `backend/models/suite.py` — Scenario, Assertion union, Verdict enum (fixes #2, #4)
3. `backend/models/world_state.py` — WorldState wrapper with patch/snapshot
4. `backend/services/dag.py` — DAG construction, batching, cascade skip (fix #3)
5. `backend/llm/client.py` — LLM abstraction for 3 providers
6. `backend/agents/orchestrator.py` — StateGraph wiring

---

## Verification

After each phase, verify by running:
```bash
uv run pytest tests/unit/ -v          # Phase 1+
uv run pytest tests/integration/ -v   # Phase 4+
docker compose up -d                  # Phase 5+
uv run pytest tests/e2e/ -v           # Phase 7
```

Final E2E: submit example YAML via UI → watch live dashboard → view report with correct verdicts.
