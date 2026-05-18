# CLAUDE.md — ATA (Agent Testing Agent)

## What this project is

ATA is a black-box end-to-end testing framework for conversational agents. It talks to an agent the same way a real user would — over HTTP or WebSocket — and verifies that the agent behaves correctly across a range of positive and negative scenarios. No access to the agent's source code is required.

The core differentiator from tools like DeepEval is the execution model: ATA is **outside-in**. It does not instrument agent internals. It connects to an endpoint, runs conversations, and evaluates outcomes. Think Playwright for conversational agents.

ATA is itself an agent — a system of six coordinated AI agents orchestrated via LangGraph, testing the agent under test from the outside.

---

## Tech stack

- **Backend:** Python, FastAPI, use uv package manager
- **Orchestration:** LangGraph (ATA's internal agent graph), langchain
- **LLM calls:** Anthropic SDK, OpenAI SDK, Google Generative AI SDK — provider and model are user-configurable per run. All three must be supported behind a common interface.
- **Task queue:** Celery + Redis (test suite runs are async jobs)
- **Database:** PostgreSQL + SQLAlchemy + Alembic
- **Blob storage:** S3-compatible (conversation trace archives, world_state audit trail)
- **Frontend:** React
- **Containerisation:** Docker Compose (Postgres, Redis, Celery worker, FastAPI, React all wired together)
- **JSON Patch:** `jsonpatch` library (RFC 6902)

---

## User input format

The user provides a single YAML file. Parsed to dict on load. Never exposed as raw JSON anywhere — not in the UI, not in reports.

```yaml
agent_under_test:
  name: "CRM Booking Assistant"
  url: "wss://crm.example.com/chat"
  protocol: websocket          # or: http
  description: >
    Books appointments for registered customers.
    Identifies customers by phone number.
    Only offers slots from the available calendar.
    Refuses unregistered callers politely.
  capabilities:
    - appointment booking
    - customer lookup
    - slot availability check
  known_limitations:
    - does not handle rescheduling
    - english and french only

world_state:
  entities:
    - id: customer_1
      phone: "+33612345678"
      name: "Isabelle Martin"
      verified: true
  catalog:
    available_slots:
      - "2026-05-20T10:00"
      - "2026-05-20T14:00"
  constraints:
    - "only verified entities can book"
    - "slots must be within 09:00-18:00"
    - "one booking per entity per day"
  context:
    current_time: "2026-05-16T08:00"
    language: "fr"
    channel: "voice"

test_config:
  total: 20
  positive: 14
  negative: 6

llm_config:
  provider: anthropic           # or: openai, google
  model: claude-sonnet-4-20250514
  # api_key set via environment variable, never in this file

# rag: key is reserved for a future release — will error if present in v1
```

**Validation rules applied during YAML parsing:**
- `test_config.total` must equal `test_config.positive + test_config.negative` — reject with clear error if mismatch
- `test_config.total` must be at least 1; `positive` and `negative` must be non-negative
- `rag` key must not be present — reject with clear error (reserved for future release)

---

## world_state schema

`world_state` has four fixed top-level keys with defined semantics. Content within each key is free-form — no enforced schema beyond the key names. The framework only cares which bucket data is in, not the internal structure.

| Key | Semantic meaning | Used by agents to... |
|-----|-----------------|----------------------|
| `entities` | Named, identifiable actors or objects the agent recognizes and looks up. Have attributes. A customer, a doctor, a user account, a document. | Build personas and lookup inputs for scenarios. Mutate attributes slightly to generate negative identity cases. |
| `catalog` | The finite set of options, items, or answers the agent can offer or act on. Appointment slots, product SKUs, policy Q&A pairs, flight options. | Pick valid values for positive scenarios. Step just outside the set for negative scenarios. |
| `constraints` | Natural language rules defining what valid behavior looks like. Not a list of values — rules about values. "Only verified entities can book." "Slots must be within business hours." | Understand the boundary of valid behavior and deliberately cross it to generate negative scenarios. |
| `context` | Read-only runtime facts. Current time, channel, language. Not tested directly. | Make generated scenarios realistic. Injected as background into agent prompts. |

**Entities are looked up. Catalog is chosen from. Constraints define the rules of the game.**

**Key rule on negative scenarios:** invalid is not enumerated — it is infinite. Everything outside `catalog` and `entities` is by definition invalid. ScenarioGeneratorAgent generates negative cases by taking valid values and mutating them slightly (wrong phone digit, slot 30 minutes off, out-of-scope question) or by deliberately violating a `constraints` rule. It does not need a list of invalid things.

`world_state` is shared and mutable across all scenarios in a suite. It evolves as scenarios run via JSON Patch. `agent_under_test` is a separate top-level key — it is not part of world_state.

---

## The full pipeline

```
user YAML
  → [OrchestratorAgent] parse + validate, load world_state
  → [ScenarioGeneratorAgent] generate all scenarios (frozen before execution starts)
  → for each scenario in DAG order:
      → [UserSimulatorAgent] run conversation turn by turn via HTTP/WS adapter
      → [ScorerAgent] determine run outcome → apply verdict flow → emit verdict
      → [WorldStatePatcherAgent] transcript → JSON Patch → validate → apply to world_state
      → store transcript + verdict + world_state diff (blob)
  → [ReporterAgent] synthesise all results → final report
```

---

## The six ATA agents

### OrchestratorAgent

Owns the LangGraph state graph. The central coordinator.

Responsibilities:
- Parse and validate the user YAML on load
- Reject with a clear error if `rag` key is present (deferred to future release)
- Hand off to ScenarioGeneratorAgent and store frozen output
- Build the scenario execution DAG from `depends_on` declarations
- Route between agents in the correct order
- Decide what to do when something breaks: halt suite, skip scenario, or flag for review
- Emit progress events to Redis pub/sub (consumed by FastAPI WebSocket → frontend)

### ScenarioGeneratorAgent

Runs once before execution starts. Output is frozen — scenarios are never regenerated mid-run.

Input: `world_state` (full) + `agent_under_test` + `test_config`

Generates N scenario objects. Each scenario has:
- `id` — unique string
- `type` — `positive` or `negative`
- `description` — what this test is checking
- `turns` — list of user utterances (may use `{{placeholders}}` — see **Placeholder resolution** below)
- `persona` — which entity from `world_state.entities` is speaking (optional)
- `assertions` — list of assertion objects (see **Assertion schema** below)
- `depends_on` — scenario id this must run after (optional, for probe chains)
- `depends_on_type` — `"probe"` (positive scenario verification) or `"defensive_probe"` (negative scenario state corruption check), only set when `depends_on` is present

Rules this agent must follow:
- Negative scenarios cross a `world_state.constraints` rule deliberately — not arbitrary nonsense
- Reads `agent_under_test.known_limitations` and does not generate scenarios for out-of-scope capabilities
- Placeholders in turns must resolve to actual values in world_state
- Probe scenarios (for state verification) are generated here and declared with `depends_on`
- Defensive probe scenarios are generated for negative scenarios that could have stateful side effects — if the agent incorrectly accepts, a defensive probe verifies state was not corrupted

Output: validated JSON array of scenario objects, stored before any scenario runs.

### Assertion schema

Three assertion types, implemented as a discriminated union on `type`:

**`world_state`** — checks a JSON pointer path in world_state changed after the scenario:
- `path` — JSON pointer (e.g. `/catalog/available_slots`)
- `operator` — `removed` | `added` | `equals` | `contains` | `not_contains`
- `expected_value` — (optional) for `equals`/`contains`/`not_contains` checks
- `description` — human-readable (e.g. "booked slot should be removed from catalog")

**`transcript`** — LLM evaluates a semantic check against the transcript:
- `check` — natural language condition (e.g. "agent mentioned a confirmation number")
- `speaker` — `"agent"` or `"user"` (default: `"agent"`)
- `description` — human-readable

**`behavioral`** — LLM classifies agent behavior:
- `expected_behavior` — `refusal` | `confirmation` | `clarification` | `escalation`
- `description` — human-readable

### UserSimulatorAgent

Plays the user in a live conversation, one turn at a time. The only ATA agent that communicates directly with the agent under test.

Stateful within a scenario — tracks full conversation history and the agent's responses to decide what to say next. Stays in character as the persona defined in the scenario. Does not evaluate or judge — only simulates.

If the agent under test goes off-script, UserSimulatorAgent responds as a real user would, then records the deviation in the transcript for ScorerAgent to evaluate.

Communicates via the protocol adapter (HTTP or WebSocket). Does not know which adapter is in use — both emit the same `Transcript` object.

### ScorerAgent

Reads the finished transcript and determines whether the conversation was a **success run** or **failure run**, then applies the verdict flow to produce the final scenario verdict.

Uses LLM to evaluate each assertion against the transcript (returns `{satisfied: bool, reasoning: str}` per assertion). `WorldStateAssertion` checks with simple operators (removed, equals) can short-circuit deterministically without LLM. The verdict flow logic itself (positive/negative routing, probe chain routing) is deterministic Python code operating on the boolean assertion results.

See **Run outcomes and verdict flow** section for the full verdict logic.

### WorldStatePatcherAgent

Runs after each scenario completes. Reads the finished transcript and emits a JSON Patch describing what changed in world_state.

Input: current `world_state` (full) + finished conversation transcript

Tool this agent calls:
```python
{
    "name": "patch_world_state",
    "description": "Apply mutations to world_state inferred from the transcript",
    "parameters": {
        "ops": {
            "type": "array",
            "items": {
                "op":    "add | remove | replace | move | copy | test",
                "path":  "/json/pointer/path",
                "value": "any JSON value (for add/replace)"
            }
        }
    }
}
```

Validation before apply — deterministic code, not LLM:
- All paths must exist in current world_state (for remove/replace/move)
- Values in `add`/`replace` must match the type of the existing field if replacing
- Values being moved must originate from a path that exists in world_state — no invented values
- If validation fails: do not apply patch, emit `PATCH_FAILED`, freeze world_state, flag for human review
- On `PATCH_FAILED`: all scenarios that transitively depend on the failed scenario (BFS on the DAG) are skipped with verdict `ERROR` and reason `"PATCH_FAILED cascade from {scenario_id}"`. The suite continues executing unaffected scenarios. World_state stays frozen at its last known good state.

If uncertain: emit empty ops array. Freezing state is always safer than a wrong mutation.

### ReporterAgent

Runs once after all scenarios complete. Synthesises all transcripts, verdicts, and world_state diffs into a final report.

Report contains:
- Overall verdict counts (SUCCESS, SUCCESS_UNVERIFIED, FAILURE, FAILURE_CORRUPT, SUSPECT, ERROR)
- Per-scenario breakdown with transcript, verdict, and evidence
- world_state audit trail (before/after each scenario)
- Failure analysis: which constraints were violated, which entity lookups failed, which catalog values were misused
- Probe chain outcomes: which primary scenarios were confirmed or marked SUSPECT

---

## Run outcomes and verdict flow

### Primitive run outcomes

Every conversation UserSimulatorAgent runs against the agent under test produces exactly one of two primitive outcomes:

- **Success run** — the agent accomplished what the scenario asked
- **Failure run** — the agent did not accomplish it

These are determined by ScorerAgent from the transcript. They are not the final verdict — the final verdict depends on scenario type and whether a probe was fired.

### Verdict flow

```
POSITIVE SCENARIO  (expected: success run)
│
├── run → failure run
│   └── verdict: FAILURE
│       no probe fired
│
└── run → success run
    │
    ├── probe not possible (no stateful side effect to verify)
    │   └── verdict: SUCCESS_UNVERIFIED
    │
    └── probe fired
        ├── probe → failure run   (agent refused — state persisted correctly)
        │   └── verdict: SUCCESS
        │
        └── probe → success run  (agent accepted — state did not persist)
            └── verdict: SUSPECT


NEGATIVE SCENARIO  (expected: failure run)
│
├── run → failure run
│   └── verdict: SUCCESS
│
└── run → success run                 (agent incorrectly accepted)
    │
    ├── no stateful side effect possible
    │   └── verdict: FAILURE
    │
    └── defensive probe fired
        ├── probe detects state corruption
        │   └── verdict: FAILURE_CORRUPT
        │
        └── probe shows state clean
            └── verdict: FAILURE


ANY SCENARIO
│
└── framework-level problem (timeout, connection failure, patch validation failed)
    └── verdict: ERROR
        does not count as a test verdict — reported separately
```

### Verdict reference

| Verdict | Scenario type | What it means |
|---------|--------------|---------------|
| `SUCCESS` | positive | Agent succeeded and probe confirmed state persisted |
| `SUCCESS_UNVERIFIED` | positive | Agent succeeded but no probe was possible |
| `SUCCESS` | negative | Agent correctly refused |
| `FAILURE` | positive | Agent failed to accomplish the scenario |
| `FAILURE` | negative | Agent accepted when it should have refused |
| `FAILURE_CORRUPT` | negative only | Agent incorrectly accepted and defensive probe confirmed state corruption |
| `SUSPECT` | positive only | Agent appeared to succeed but probe disproved state persistence |
| `ERROR` | any | Framework could not complete the run |

### Probe scenarios

State verification never reads a database directly. A probe is a second conversation fired after a positive scenario produces a success run:

- Scenario A (positive): books slot X → success run → verdict pending
- Scenario B (probe, `depends_on: scenario_a`): tries to book slot X again → should produce failure run

If probe → failure run: Scenario A verdict = `SUCCESS`
If probe → success run: Scenario A verdict = `SUSPECT` (state did not hold)

Probe scenarios are generated by ScenarioGeneratorAgent and declared with `depends_on` and `depends_on_type: probe`. OrchestratorAgent routes them after their parent scenario completes. Defensive probes for negative scenarios use `depends_on_type: defensive_probe`.

---

## Protocol adapters

The scenario schema is protocol-agnostic. Adapter selected from `agent_under_test.protocol`.

**HTTP adapter:** stateless POST per turn. Session state managed via cookie or session header returned by agent.

**WebSocket adapter:** one connection per scenario, turns sent sequentially, connection closed on scenario end.

Both adapters emit the same `Transcript` object. UserSimulatorAgent does not know which adapter is in use. A non-response from the agent under test produces verdict `ERROR` — not a framework crash.

---

## Placeholder resolution

`{{path/to/value}}` placeholders in scenario turns are resolved at **execution time** (not generation time) against the **current** world_state. This means placeholders reflect any mutations from earlier scenarios.

Resolution uses JSON pointer: `{{entities/0/phone}}` resolves to the value at `/entities/0/phone` in world_state.

If a placeholder path does not resolve, the scenario receives verdict `ERROR` with reason `"unresolved placeholder: {{path}} — path not found in current world_state"`. The scenario is not retried.

---

## Scenario execution model

Independent scenarios (no `depends_on` relationship) run in parallel within a topological batch via `asyncio.gather`. World_state is read-only during a batch — mutations from WorldStatePatcherAgent are applied sequentially after the batch completes. The next batch sees the updated world_state.

---

## What is deterministic (no LLM)

- Applying and validating JSON Patch to world_state
- Evaluating `world_state` assertions with simple operators (removed, equals, contains)
- Executing conversation turns (send/receive via adapter)
- Building the execution DAG from `depends_on` declarations
- Validating scenario schema after generation
- Parsing and validating user YAML on load
- Applying the verdict flow logic (routing based on scenario type + assertion results)
- Placeholder resolution
- PATCH_FAILED cascade propagation

## What uses LLM (inside agents)

- **ScenarioGeneratorAgent** — generating scenario objects from world_state
- **UserSimulatorAgent** — generating each user turn in character
- **ScorerAgent** — evaluating transcript and behavioral assertions against conversation transcripts
- **WorldStatePatcherAgent** — inferring world_state mutations from transcript via tool call
- **ReporterAgent** — synthesising failure analysis narrative

---

## Project structure

```
/
├── backend/
│   ├── main.py                        # FastAPI app
│   ├── models/
│   │   ├── suite.py                   # Suite, Scenario, Assertion, ScenarioVerdict
│   │   ├── world_state.py             # WorldState wrapper + patch application
│   │   └── transcript.py             # Turn, Transcript
│   ├── agents/
│   │   ├── orchestrator.py           # OrchestratorAgent — LangGraph graph definition
│   │   ├── scenario_generator.py     # ScenarioGeneratorAgent
│   │   ├── user_simulator.py         # UserSimulatorAgent
│   │   ├── world_state_patcher.py    # WorldStatePatcherAgent
│   │   ├── scorer.py                 # ScorerAgent
│   │   └── reporter.py               # ReporterAgent
│   ├── adapters/
│   │   ├── base.py                   # Adapter interface + Transcript model
│   │   ├── http.py                   # HTTP protocol adapter
│   │   └── websocket.py              # WebSocket protocol adapter
│   ├── llm/
│   │   └── client.py                 # Common LLM interface — Anthropic, OpenAI, Google
│   ├── rag/
│   │   └── README.md                 # RAG support deferred to future release
│   ├── tasks.py                      # Celery task — instantiates and runs OrchestratorAgent
│   └── db/
│       ├── models.py                 # SQLAlchemy ORM models
│       └── migrations/               # Alembic
├── frontend/
│   └── src/
│       ├── pages/
│       │   ├── NewRun.jsx            # YAML input + LLM provider/model config
│       │   ├── RunDashboard.jsx      # live run progress via WebSocket
│       │   └── Report.jsx            # results + failure analysis
│       └── components/
├── docker-compose.yml
├── CLAUDE.md                         # this file
└── README.md
```

---

## Key implementation notes for Claude Code

- Start with `models/world_state.py` and `models/suite.py` — everything else depends on these
- `WorldState` is a wrapper around a plain dict. Patch application uses `jsonpatch.apply_patch`. Always snapshot before applying. Snapshots stored in blob storage, not Postgres. Postgres stores only run metadata and final verdicts.
- The LangGraph graph is defined in `agents/orchestrator.py`. Each of the other five agents is a node. Conditional edges handle probe routing and failure modes.
- The Celery task in `tasks.py` instantiates and runs OrchestratorAgent for one suite. FastAPI creates the task and returns a run ID. Frontend subscribes via WebSocket for progress — OrchestratorAgent emits events to Redis pub/sub.
- All LLM calls go through `llm/client.py` — a single common interface that supports Anthropic, OpenAI, and Google. Provider and model are read from `llm_config` in the user YAML. API keys are always set via environment variables, never stored in the database or YAML.
- ScenarioGeneratorAgent runs once at suite start. Output is stored and frozen. Do not regenerate scenarios mid-run under any circumstance.
- WorldStatePatcherAgent input is transcript + current world_state only. Keep context minimal — do not pass full suite history.
- Protocol adapters must handle reconnection and timeout gracefully. TIMEOUT produces verdict ERROR — it is not an exception.
- User YAML is parsed with `yaml.safe_load()` on load and validated immediately. Never re-serialised to JSON for display anywhere in the UI.

---

## What ATA is NOT

- Not a metric library
- Not an agent observability tool (no instrumentation, no decorators, no SDK wrappers)
- Not a load testing tool (concurrency is for running multiple independent suites, not hammering one agent)
- Not a prompt evaluation framework (tests agent behavior over conversations, not single prompt/response pairs)

## Git & Workflow Constraints
- DO NOT attempt to run git commit, push, or reset commands. 
- When your code changes are ready, tell the user exactly what changes were made and ask them to perform the commit manually.