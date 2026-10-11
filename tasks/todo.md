# Tasks: voice-run

Dependency-ordered. Each task ≤5 files, with acceptance + verification. See [plan.md](./plan.md).

- [x] **T1 — Switch platform dependency to `agentspect[pipecat]` (thick voice)**
  - Acceptance: `pyproject.toml` requires `agentspect[pipecat] @ git+…`; lockfile updated; the pipecat/Silero voice stack imports.
  - Verify: `uv lock && uv sync`; `uv run python -c "import agentspect, pipecat; from agentspect.voice import create_voice_io; print('ok')"`.
  - Files: `pyproject.toml`, `uv.lock`.

- [x] **T2 — Add `voice_websocket` to the protocol enum + Alembic migration**
  - Acceptance: `ProtocolDB.VOICE_WEBSOCKET` exists; a migration adds the value to the PG type; existing rows intact.
  - Verify: `uv run alembic upgrade head` on a populated DB; insert a `Suite(protocol=voice_websocket)` succeeds (quick script or the T7 test).
  - Files: `backend/db/models.py`, `backend/db/migrations/versions/<rev>_add_voice_websocket.py`.
  - Notes: autocommit the `ALTER TYPE … ADD VALUE IF NOT EXISTS`; downgrade is a documented no-op.

- [x] **T3 — Raise Celery soft-timeout for voice suites**
  - Acceptance: voice suites get a higher soft/hard time limit than text so a multi-minute real-time run isn't reaped; value configurable in `config.py`.
  - Verify: config present and applied in the task path; a voice suite's effective limit reflects it (assert in the task wiring / a small unit check).
  - Files: `backend/config.py`, `backend/tasks.py` (or `backend/celery_app.py`).

- [x] **T4 — Pre-enqueue voice key guard in the run endpoint**
  - Acceptance: POSTing a `voice_websocket` suite with no configured STT/TTS provider key returns `400` with a clear message, before any Celery job is created.
  - Verify: unit/async test asserts 400 + message when the key env is unset; a normal voice POST with keys present still enqueues.
  - Files: `backend/main.py`, `backend/config.py` (helper to resolve a provider's key).

- [x] **T5 — Surface voice telemetry + metrics through the API responses**
  - Acceptance: run/report/transcript responses expose each turn's `voice`/`voice_events` and the report's voice metrics (no fields stripped); telemetry stays in `turns_json` (no new columns).
  - Verify: for a stored voice run, `GET /api/runs/{id}/scenarios/{sid}/transcript` includes a `voice` block and `GET …/report` includes voice metrics (asserted in T7).
  - Files: `backend/schemas/*.py` (and `backend/main.py` response mapping if it drops fields).

- [x] **T6 — In-repo mock voice agent (voice-WS frame protocol)**
  - Acceptance: a FastAPI WS/ASGI mock that receives `{"type":"audio",…}` frames, STTs them (agentspect `VoiceIO`/OpenAI), runs minimal agent logic (reuse the text mock's loop), TTSs replies, and streams audio frames back + `end_of_speech`.
  - Verify: a standalone smoke check drives one turn through the mock and gets audio back; fully exercised by T7.
  - Files: `tests/e2e/voice_mock_agent.py` (+ optional small helper).
  - Notes: user supplies required provider keys; keep utterances short.

- [x] **T7 — Integration test: full voice_websocket run (capstone)**
  - Acceptance: submits a tiny (`total: 1`) `voice_websocket` suite against T6, injected via the engine adapter patch point (`agentspect.agents.orchestrator.create_adapter`), through the real API→Celery-body→DB/S3→API path; produces a verdict; a persisted turn has a non-null `voice` block; report includes voice metrics. Infra-gated (skips without Postgres/Redis/MinIO + keys).
  - Verify: `uv run pytest tests/integration/test_voice_run.py -v` passes with the stack up; existing text tests stay green.
  - Files: `tests/integration/test_voice_run.py`.
