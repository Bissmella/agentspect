# Spec: voice-run

Module `voice-run` of the [Voice Channel capability map](./CAPABILITY-MAP-voice.md).
First module in build order; everything else (telephony, recording, UI) builds on it.

## Objective

Let the ATA **platform** accept, run, persist, and serve a `voice_websocket` test
suite end to end — so a user submits a YAML with `protocol: voice_websocket` and an
optional `voice:` block and gets back voice-aware results exactly as they do for text
runs today. The engine already runs voice conversations; this module is the platform
plumbing that was never built around that capability.

**User story:** As a tester, I POST a `voice_websocket` or agent's telephone number suite (agent reachable over the
ATA voice-WS frame protocol or telephone number) and the platform runs it, stores each turn's
`VoiceMeta`/`voice_events`, and serves voice metrics in the report — without me doing
anything different from a text run.

**Non-goals (other modules):** telephony/Telnyx, audio recording to S3, and any UI.
This module targets a **direct voice-WS agent** (the engine's existing transport); the
`telephony` module later swaps in a Telnyx-backed connection.

**Success looks like:** an integration test submits a `voice_websocket` suite against a
mock voice agent and it runs green through the real platform pipeline (API → Celery
task → engine → Postgres/S3 → API reads), with voice telemetry persisted and
retrievable.

## Tech Stack

- Platform: FastAPI, Celery + Redis, PostgreSQL + SQLAlchemy + Alembic, S3/MinIO, `uv`.
- Engine: `agentspect[pipecat]` from git — **thick voice**: Silero VAD endpointing
  (`endpointing.mode: vad`) plus OpenAI STT/TTS. The platform dependency changes from
  `agentspect @ git+…` to `agentspect[pipecat] @ git+…` (pulls Silero; heavier image).
- Voice provider keys reused from existing env (`OPENAI_API_KEY` already in `config.py`).

## Commands

```
Migrate:  uv run alembic upgrade head
New rev:  uv run alembic revision -m "add voice_websocket protocol"
Test:     uv run pytest tests/integration/test_voice_run.py -v
Lint:     uv run ruff check backend
Stack:    docker compose up -d postgres redis minio   # Postgres on :5433 (override)
```

## Project Structure

Files this module touches (all platform-side):

```
pyproject.toml                          → dep: agentspect → agentspect[pipecat] (thick voice)
backend/db/models.py                    → add ProtocolDB.VOICE_WEBSOCKET
backend/db/migrations/versions/*.py     → new Alembic rev: ALTER TYPE … ADD VALUE
backend/config.py                       → raised Celery soft-timeout for voice suites
backend/main.py                         → pre-enqueue guard: required STT/TTS key present (explicit error)
backend/schemas/*.py                    → surface VoiceMeta/voice_events + voice metrics in run/report responses
tests/integration/test_voice_run.py     → new; + a mock voice-WS agent fixture
```

No new top-level packages. `tasks.py` is expected to need **no change** — it already
stores `sc["turns"]` as JSONB, and voice turns carry `voice`/`voice_events` inline, so
they persist for free. Confirm during implementation.

## Code Style

Match the existing codebase: typed SQLAlchemy `Mapped[...]`, explicit error handling
with clear messages (per project convention), short "why" comments.

```python
# db/models.py — Postgres enum gains a value; existing rows unaffected.
class ProtocolDB(str, PyEnum):
    HTTP = "http"
    WEBSOCKET = "websocket"
    VOICE_WEBSOCKET = "voice_websocket"  # engine already validates it; platform must store it
```

```python
# Postgres enums can't ADD VALUE inside a transaction — autocommit the alter.
def upgrade() -> None:
    op.execute("COMMIT")
    op.execute("ALTER TYPE protocoldb ADD VALUE IF NOT EXISTS 'voice_websocket'")
```

```python
# main.py — fail fast with a clear message, not an opaque mid-run error.
if yaml_input.agent_under_test.protocol == "voice_websocket":
    provider = (yaml_input.voice or VoiceConfig()).stt.provider
    if not _voice_key_for(provider):
        raise HTTPException(400, detail=f"voice run needs an API key for STT provider '{provider}'")
```

## Testing Strategy

- **Framework:** pytest + pytest-asyncio (existing). Live-infra gated like the current
  e2e test (skip unless Postgres/Redis/MinIO + key present).
- **Integration (`tests/integration/test_voice_run.py`):** a mock voice agent that
  speaks the ATA voice-WS JSON frame protocol (`{"type":"audio","data":<b64 wav>}`,
  `{"type":"end_of_speech"}`), injected via the engine's adapter patch point (as the
  existing text mock does with `create_adapter`). Assert: the suite runs, verdicts are
  produced, and at least one stored turn carries a non-null `voice` block.
- **Unit:** the migration applies and `voice_websocket` inserts succeed; the main.py
  key-guard returns 400 when the STT key is missing.
- One runnable check is enough per the project's bar — no exhaustive per-function suite.

## Boundaries

- **Always:** run `alembic upgrade head` + the integration test before calling this done;
  keep voice telemetry in existing JSONB (`turns_json`) unless a query need proves otherwise;
  explicit 400s with actionable messages.
- **Ask first:** adding the `agentspect[pipecat]` extra (changes the dependency + image);
  adding new DB columns vs. staying in JSON; any new third-party dependency.
- **Never:** patch/fork agentspect locally; commit API keys; weaken or delete the existing
  text-run tests to make voice pass.

## Success Criteria

1. A YAML with `protocol: voice_websocket` POSTed to `/api/runs` is accepted (no enum
   error) and a `Suite` row persists with `protocol = voice_websocket`.
2. `alembic upgrade head` on an existing DB adds the enum value without data loss.
3. The integration test runs a full voice suite against the mock voice agent and passes.
4. At least one persisted scenario turn exposes `VoiceMeta` (e.g. `time_to_first_audio_ms`)
   through the run/report API responses.
5. Submitting a voice suite with no STT/TTS key configured returns a clear 400 before any
   Celery job is enqueued.
6. Existing text-run tests remain green.

## Resolved Decisions

1. **Thick voice.** Use `agentspect[pipecat]` with Silero VAD endpointing — not thin.
   Accepts the heavier image as the cost of real speech-boundary detection.
2. **Telemetry storage in JSONB.** Keep `VoiceMeta`/`voice_events` nested inside the
   existing `turns_json` — no new columns, no migration for telemetry. Promote specific
   fields to columns later only if a cross-run query/sort need appears.
3. **In-repo mock voice agent.** Build a minimal mock that speaks the voice-WS frame
   protocol, injected via the engine adapter patch point (mirrors the existing text mock).
   User provides any required provider APIs/keys.
4. **Timeouts.** Per-turn ceiling comes from the engine/`voice` endpointing defaults
   (`silence_ms`); raise the Celery soft-timeout for voice suites so a multi-minute
   real-time run isn't reaped. Concrete numbers chosen during Plan.
