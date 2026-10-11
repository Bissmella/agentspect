# Plan: voice-run

Implementation plan for the `voice-run` module ([spec](./SPEC-voice-run.md),
[map](./CAPABILITY-MAP-voice.md)). Reviewed before any code.

## Key finding — no engine change needed

The engine already runs voice end to end from the YAML: `create_adapter` routes
`voice_websocket → VoiceWebSocketAdapter`, the orchestrator passes `voice=yaml_input.voice`
through, and voice metrics (`time_to_first_audio`, `stt_confidence`, silence) are registered
and read `turn.voice`. The platform passes the raw YAML straight to `OrchestratorAgent`, so
voice config and voice telemetry already flow through. `voice-run` is therefore a thin set of
platform edits: let the protocol in, fail fast without keys, don't strip voice fields on the
way out, and prove it.

## Components & dependency order

```
T1 dep: agentspect[pipecat]  ──┐ (must land first — installs the voice/VAD stack)
                               ├──► T6 mock voice agent ──► T7 integration test (capstone)
T2 DB enum + migration  ───────┤
T3 Celery voice timeout ───────┤ (T2/T3/T4/T5 are independent small edits — parallel)
T4 main.py key guard   ────────┤
T5 schema pass-through ────────┘
```

**Vertical slices (each leaves the app runnable):**
1. **Accept & run** — T1 + T2: a `voice_websocket` suite can be POSTed and executed.
2. **Safe & visible** — T4 + T3 + T5: fail-fast on missing keys, no mid-run reaping, voice telemetry surfaced in API.
3. **Proven** — T6 + T7: mock voice agent + green integration test.

**Sequential:** T1 before T6/T7. T6 before T7. **Parallel:** T2, T3, T4, T5.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| `pipecat[silero]` is heavy (pulls torch); may be slow/awkward to install on Windows | T1 gates everything — verify `uv sync` + import immediately; if torch wheels fail, use CPU-only index; note the Docker image grows (Dockerfile bump tracked, not blocking) |
| Postgres `ALTER TYPE … ADD VALUE` can't run inside a transaction | Migration autocommits (`op.execute("COMMIT")`) then `ADD VALUE IF NOT EXISTS`; downgrade is a documented no-op (PG can't drop an enum value) |
| Integration test makes real OpenAI STT/TTS calls — cost, flakiness, needs keys | Infra-gate the test (skip without keys, like the current e2e); keep suite tiny (`total: 1`), short utterances, temperature 0 |
| Mock voice agent is itself a voice agent (must STT in / TTS out) | Reuse agentspect `VoiceIO` (OpenAI) inside the mock; keep its logic minimal (reuse the text mock's tool/LLM loop), wrap it in a voice-WS frame endpoint |
| Python enum + SQLAlchemy `Enum` must agree | T2 updates the `ProtocolDB` Python enum *and* the DB type in one task |

## Verification checkpoints

- **After T1:** `uv lock` resolves; `uv run python -c "import agentspect, pipecat"` succeeds.
- **After T2:** `alembic upgrade head` on a DB with existing rows; a `voice_websocket` Suite inserts without enum error.
- **After T4:** unit test — POST a voice suite with no STT key → 400 before any Celery enqueue.
- **After T7:** integration test green; a persisted turn carries a non-null `voice` block; the report summary includes voice metrics.

## Out of scope (other modules)

Telnyx / telephony / phone-number origination (`telephony`), audio-to-S3 recording
(`recording`), and all UI (`voice-ui`). This module targets the direct voice-WS agent only.
