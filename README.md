# ATA — Agent Testing Agent

**Playwright for conversational agents.** ATA is a black-box, end-to-end testing
framework for conversational agents. It talks to an agent the same way a real user
would — over HTTP or WebSocket — and verifies that the agent behaves correctly
across a range of positive and negative scenarios. No access to the agent's source
code is required.

The core differentiator from tools like DeepEval is the execution model: ATA is
**outside-in**. It does not instrument agent internals. It connects to an endpoint,
runs conversations, and evaluates outcomes. ATA is itself an agent — a system of six
coordinated AI agents orchestrated via LangGraph, testing the agent under test from
the outside.

---

## How it works

You provide a single YAML file describing the agent under test, a `world_state`
(the entities it knows, the catalog it can offer, the rules it must follow), how
many test scenarios to run, and which LLM to drive ATA. ATA then:

1. **Generates** a frozen set of positive and negative scenarios from your world_state.
2. **Runs** each scenario as a live conversation against your agent's endpoint.
3. **Scores** each transcript against its assertions and applies a verdict flow.
4. **Probes** stateful successes with a follow-up conversation to confirm the state
   really persisted (and defensive probes to detect corruption on incorrect accepts).
5. **Patches** world_state from each transcript so later scenarios see the changes.
6. **Reports** quantitative metrics, per-scenario breakdowns, and a failure analysis.

```
user YAML
  → OrchestratorAgent      parse + validate, load world_state
  → ScenarioGeneratorAgent generate all scenarios (frozen)
  → for each scenario in DAG order:
      → UserSimulatorAgent      run the conversation via HTTP/WS
      → ScorerAgent             evaluate assertions → verdict
      → WorldStatePatcherAgent  transcript → JSON Patch → apply
  → ReporterAgent          synthesise the final report
```

The report quantifies six metrics: **task completion**, **boundary adherence**,
**state verification** (probe-confirmed persistence), **per-constraint violation
breakdown**, **recovery behavior** (clean refusal vs. confused/error/leak), and
**conversation efficiency** (avg turns to completion / refusal).

---

## Tech stack

- **Backend:** Python 3.12, FastAPI, [uv](https://docs.astral.sh/uv/)
- **Orchestration:** LangGraph + LangChain (ATA's internal agent graph)
- **LLMs:** Anthropic, OpenAI, Google, OpenRouter, Ollama behind one common interface
- **Task queue:** Celery + Redis (suite runs are async jobs)
- **Database:** PostgreSQL + SQLAlchemy + Alembic
- **Blob storage:** S3-compatible (MinIO locally) for transcripts and world_state audit
- **Frontend:** React + Vite + Tailwind + TanStack Query
- **Everything wired together via Docker Compose**

---

## Quick start (Docker Compose)

The fastest path to a running system — Postgres, Redis, MinIO, the API, a Celery
worker, and the React frontend all wired together.

```bash
# 1. Configure environment
cp .env.example .env
#    Edit .env and set the API key for the provider you'll use, e.g.:
#    ANTHROPIC_API_KEY=sk-ant-...

# 2. Apply database migrations (one-off, profile-gated service)
docker compose run --rm migrate

# 3. Bring the stack up
docker compose up -d

# 4. Open the UI
#    Frontend:  http://localhost:3000
#    API docs:  http://localhost:8000/docs
#    MinIO:     http://localhost:9001  (minioadmin / minioadmin)
```

In the UI, open **New Run**, paste one of the files from [`examples/`](examples/)
(or the pre-filled default), and click **Run Tests**. Watch the live dashboard, then
open the report.

---

## Local development (without Docker for the app)

Run the infrastructure in containers but the backend and frontend on your host for
faster iteration.

```bash
# Infra only
docker compose up -d postgres redis minio

# Backend deps + migrations
uv sync
uv run alembic upgrade head

# API server
uv run uvicorn backend.main:app --reload

# Celery worker (separate terminal)
uv run celery -A backend.celery_app worker --loglevel=info

# Frontend (separate terminal)
cd frontend
npm install
npm run dev            # Vite dev server, proxies /api to the backend
```

---

## The input YAML

A single YAML file drives a run. See [`examples/booking_agent.yaml`](examples/booking_agent.yaml)
(a stateful WebSocket booking agent) and [`examples/faq_agent.yaml`](examples/faq_agent.yaml)
(a stateless HTTP FAQ agent) for complete, working examples.

```yaml
agent_under_test:
  name: "CRM Booking Assistant"
  url: "wss://crm.example.com/chat"
  protocol: websocket          # or: http
  description: "Books appointments for registered customers..."
  capabilities: [appointment booking, customer lookup]
  known_limitations: [does not handle rescheduling]

world_state:
  entities:                    # actors the agent looks up (have attributes)
    - id: customer_1
      phone: "+33612345678"
      verified: true
  catalog:                     # the finite set it can offer / act on
    available_slots: ["2026-05-20T10:00", "2026-05-20T14:00"]
  constraints:                 # natural-language rules ATA deliberately crosses
    - "only verified entities can book"
  context:                     # read-only runtime facts
    current_time: "2026-05-16T08:00"
    language: "fr"

test_config:
  total: 20                    # must equal positive + negative
  positive: 14
  negative: 6

llm_config:
  provider: anthropic          # anthropic | openai | google | openrouter | ollama
  model: claude-sonnet-4-20250514
  # API keys come from the environment, never from this file.
```

Validation is strict: `total` must equal `positive + negative`, `total >= 1`, and
the reserved `rag` key must not be present (deferred to a future release).

---

## API

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/runs` | Submit YAML, validate, enqueue a run, return `suite_id` |
| GET | `/api/runs/{id}` | Suite status + scenario verdicts |
| GET | `/api/runs/{id}/report` | Final report with quantitative metrics |
| GET | `/api/runs/{id}/scenarios` | All scenarios with verdicts |
| GET | `/api/runs/{id}/scenarios/{sid}/transcript` | Transcript from blob storage |
| WS | `/api/runs/{id}/ws` | Live progress events (Redis pub/sub) |

---

## Testing

```bash
uv run pytest tests/unit/ -v          # fast, no infrastructure needed
uv run pytest tests/integration/ -v   # orchestrator against mock agents (needs an LLM key)
uv run pytest tests/e2e/ -v           # full stack — see below
```

The end-to-end test (`tests/e2e/test_full_run.py`) exercises the real API, Postgres,
Redis, and S3 together. It **skips automatically** unless that infrastructure is
reachable and an LLM key is set, so a plain `uv run pytest` stays green with nothing
running. To run it for real:

```bash
docker compose up -d postgres redis minio
export OPENROUTER_API_KEY=...          # or set it in .env
uv run pytest tests/e2e/test_full_run.py -v
```

---

## Project layout

```
backend/
  main.py              FastAPI app + routes
  models/              domain models (world_state, suite, transcript, yaml_input)
  agents/              the six ATA agents + LangGraph orchestrator + graph state
  adapters/            HTTP and WebSocket protocol adapters
  llm/                 common LLM interface (Anthropic/OpenAI/Google/OpenRouter/Ollama)
  services/            yaml parsing, placeholders, DAG, blob storage
  metrics.py           the six quantitative report metrics
  tasks.py             Celery task that runs one suite
  db/                  SQLAlchemy models + Alembic migrations
frontend/src/
  pages/               NewRun, RunDashboard, Report
  components/          YAML editor, metrics dashboard, transcript viewer, ...
examples/              ready-to-run YAML inputs
tests/                 unit, integration, e2e
```

See [`CLAUDE.md`](CLAUDE.md) for the full architecture specification.

---

## What ATA is NOT

- Not a metric library, not an agent observability tool (no instrumentation/SDK wrappers).
- Not a load-testing tool (concurrency is for running independent suites, not hammering one agent).
- Not a prompt-evaluation framework (it tests behavior over conversations, not single prompt/response pairs).
