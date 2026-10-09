# Agentspect — Agent Testing Agent

![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Status](https://img.shields.io/badge/status-alpha-orange)

**Playwright for conversational agents.** Agentspect is a black-box, **outside-in** testing
framework for conversational agents. You describe your agent's world in a YAML file; Agentspect
**writes the tests for you**, holds real conversations with the agent over its actual
interface — HTTP, WebSocket, or a plain Python callable — and reports, quantitatively,
how it behaved.

No access to the agent's source, prompts, or internals is required. Unlike
instrumentation-based eval tools, Agentspect never wraps your agent in decorators or SDKs — it
only ever observes what goes in and what comes out. Agentspect is itself an agent: a system of
six coordinated LLM agents orchestrated with LangGraph, testing your agent from the outside.


---

## Contents

- [Why Agentspect](#why-agentspect)
- [Install](#install)
- [Quick start](#quick-start)
- [Core concepts](#core-concepts)
  - [world_state — describe, don't script](#world_state--describe-dont-script)
  - [Verdicts & probes](#verdicts--probes)
  - [Assertions](#assertions)
- [How it works](#how-it-works)
- [Metrics](#metrics)
- [The input YAML](#the-input-yaml)
- [Assets](#assets)
- [The report](#the-report)
- [Public API](#public-api)
- [Configuration](#configuration)
- [Development](#development)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [What Agentspect is not](#what-agentspect-is-not)
- [Contributing](#contributing)
- [License](#license)

---

## Why Agentspect

Most agent-eval tools score single prompt/response pairs, or require you to instrument
your agent's internals. Real agents fail in ways that only show up over a *conversation*
and only from the *outside*: they double-book a slot, confidently claim they did
something they didn't, leak internal data while refusing, or accept a request they should
have declined.

Agentspect is built for exactly that. You give it a description of the world your agent operates
in — who it knows, what it can offer, the rules it must follow — and it:

- **generates** a suite of positive and negative test conversations for you,
- **runs** them turn by turn against the live agent,
- **verifies persisted state** by probing (did the booking actually stick?),
- and **quantifies** the results across task completion, boundary adherence, state
  integrity, recovery quality, latency, and more.

---

## Install

```bash
git clone https://github.com/Bissmella/agentspect
cd agentspect
uv sync                  # or:  pip install -e .
```

Requires Python 3.12+. Set the API key for whichever provider drives Agentspect:

```bash
export ANTHROPIC_API_KEY=sk-ant-...    # or OPENAI_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY
```

---

## Quick start

Describe your agent and its world in one YAML file (see [`examples/`](examples/) for
complete ones), then run it:

```python
import asyncio
from agentspect import run_suite

report = asyncio.run(run_suite(open("examples/booking_agent.yaml").read()))

print(report["verdict_counts"])   # {'success': 4, 'success_unverified': 3, 'failure': 1, ...}
print(report["metrics"]["task_completion"])
```

`run_suite` runs the entire pipeline in-process — no database, no broker, no external
services — and returns a plain report dict you can serialize, assert on in a test, or
render however you like.

### Testing a Python callable directly

No server to stand up. If your agent is a Python function or object, hand it straight to
Agentspect: set `protocol: callable` in the YAML (no `url` needed) and pass it as `agent=`. It
may be sync or async, and accept either `(message)` or `(message, history)`, where
`history` is the prior turns as `{"user": ..., "agent": ...}` dicts:

```python
async def my_agent(message: str, history: list[dict]) -> str:
    # your LangGraph graph, LangChain agent, raw LLM call — anything
    return await my_graph.ainvoke(message, history)

report = asyncio.run(run_suite(open("config.yaml").read(), agent=my_agent))
```

Still fully black-box — Agentspect only sees what the callable returns for each message.

---

## Core concepts

### world_state — describe, don't script

You never hand-write test cases. Instead you describe the **world_state**, and Agentspect's
ScenarioGeneratorAgent derives the tests from it. It has four buckets:

| Bucket | Meaning | Agentspect uses it to… |
|--------|---------|-----------------|
| `entities` | Named actors the agent looks up (a customer, an account) — they have attributes. | Build personas and lookups; mutate attributes for negative identity cases. |
| `catalog` | The finite set the agent can offer or act on (slots, SKUs, Q&A pairs). | Pick valid values for positives; step just outside the set for negatives. |
| `constraints` | Natural-language rules that define valid behavior ("only verified entities can book"). | Deliberately cross a rule to generate a negative scenario. |
| `context` | Read-only runtime facts (current time, language, channel). | Make scenarios realistic; injected as background. |

The guiding idea: **valid is enumerated, invalid is infinite.** Agentspect generates negatives
by taking valid values and nudging them just outside the boundary, or by crossing a
constraint — it never needs you to list "bad" inputs.

world_state is **shared and mutable** across a suite: it evolves as scenarios run (via
RFC 6902 JSON Patch), so later scenarios see what earlier ones changed.

### Verdicts & probes

A run doesn't just pass or fail. Agentspect determines whether the conversation was a *success
run* or *failure run*, then — crucially — **verifies persisted state by probing**: after
a positive scenario books a slot, a probe scenario tries to book it again and should be
refused. If the probe *succeeds*, the state never actually persisted, and the original
verdict becomes `SUSPECT`. Negative scenarios that are wrongly accepted get a *defensive
probe* to check whether state was corrupted.

| Verdict | Scenario | Meaning |
|---------|----------|---------|
| `SUCCESS` | positive | Succeeded, and a probe confirmed the state persisted. |
| `SUCCESS_UNVERIFIED` | positive | Succeeded, but there was no stateful effect to probe. |
| `SUSPECT` | positive | Looked successful, but a probe proved the state didn't hold. |
| `FAILURE` | positive | Did not accomplish the task. |
| `SUCCESS` | negative | Correctly refused. |
| `FAILURE` | negative | Wrongly accepted a request it should have refused. |
| `FAILURE_CORRUPT` | negative | Wrongly accepted **and** a defensive probe confirmed state corruption. |
| `ERROR` | any | Framework couldn't complete the run (timeout, connection, bad patch). Reported separately. |

### Assertions

Agentspect generates three kinds of assertion per scenario (you read them in the report; you
don't write them). Simple `world_state` checks are evaluated deterministically; the rest
use the LLM:

- **`world_state`** — a JSON-pointer path changed as expected (`operator`:
  `removed | added | equals | contains | not_contains`).
- **`transcript`** — an LLM checks a natural-language condition against the transcript
  ("the agent gave a confirmation number").
- **`behavioral`** — an LLM classifies the agent's behavior as
  `refusal | confirmation | clarification | escalation`.

---

## How it works

```
user YAML
  → OrchestratorAgent       parse + validate, load world_state, build the DAG
  → ScenarioGeneratorAgent  generate all scenarios (frozen before execution starts)
  → for each scenario in dependency order:
      → UserSimulatorAgent      hold the conversation, turn by turn, via the adapter
      → ScorerAgent             evaluate assertions → apply the verdict flow
      → WorldStatePatcherAgent  infer state changes → validate → apply (JSON Patch)
  → ReporterAgent           synthesise metrics + failure analysis
```

| Agent | Role |
|-------|------|
| **OrchestratorAgent** | Owns the LangGraph state graph; parses/validates input, routes between agents, builds the probe DAG, handles failures. |
| **ScenarioGeneratorAgent** | Runs once. Turns world_state into a frozen set of positive/negative scenarios, plus probes and defensive probes. |
| **UserSimulatorAgent** | The only agent that talks to the agent under test. Stays in persona, resolves `{{placeholders}}` at execution time, records the transcript. |
| **ScorerAgent** | Evaluates each assertion, classifies recovery quality, applies the deterministic verdict flow. |
| **WorldStatePatcherAgent** | Infers world_state mutations from the transcript and applies them as validated JSON Patch (or freezes state on failure). |
| **ReporterAgent** | Aggregates verdicts, metrics, probe outcomes, and an LLM failure analysis into the final report. |

Scenarios reference world_state values with `{{json/pointer}}` placeholders that resolve
at **execution time** against the *current* world_state — so they reflect earlier
mutations.

---

## Metrics

The report quantifies the run instead of handing back a vague pass/fail:

| Metric | Question it answers |
|--------|---------------------|
| **Task completion** | Did it do what it was supposed to do? |
| **Boundary adherence** | Does it correctly refuse what it should refuse? |
| **Verification rate** | When it says it did something, did the state actually change? |
| **Constraint violations** | Which rules does it break the most? |
| **Recovery behavior** | On adversarial input: clean refusal vs. confused / error / info-leak? |
| **Conversation efficiency** | Avg turns to completion / to refusal? |
| **Latency** | Response time per turn — avg / p50 / p95 / p99 (ms). |
| **Turn error rate** | Fraction of turns that errored at the transport layer. |

Metrics are **pluggable**. Each is a `Metric` subclass in a registry that can hook into
the run's lifecycle — `on_suite_start`, `on_scenario_start`, `on_turn`,
`on_scenario_end` — before producing its result in `compute`. Latency, for example, is a
hook metric that observes every turn.

### Writing a custom metric

```python
from agentspect import Metric, register

@register
class AvgResponseLength(Metric):
    name = "avg_response_length"
    description = "Average agent response length in characters."

    def __init__(self):
        self._lengths = []

    def on_turn(self, ctx, scenario, turn, index):      # hook into each turn
        if turn.error is None:
            self._lengths.append(len(turn.agent_response))

    def compute(self, ctx):
        n = len(self._lengths)
        return {"avg_chars": sum(self._lengths) / n if n else None}
```

Registered metrics run automatically and appear in `report["metrics"]`. Register into the
global `registry`, or build an isolated `MetricRegistry` and pass it to `MetricEngine`.
`compute_metrics(..., metrics=["latency"])` runs a chosen subset.

---

## The input YAML

A single YAML file drives a run. See [`examples/booking_agent.yaml`](examples/booking_agent.yaml)
(a stateful WebSocket booking agent) and [`examples/faq_agent.yaml`](examples/faq_agent.yaml)
(a stateless HTTP FAQ agent).

```yaml
agent_under_test:
  name: "CRM Booking Assistant"
  url: "wss://crm.example.com/chat"    # omit for protocol: callable
  protocol: websocket                  # http | websocket | callable
  description: "Books appointments for registered customers..."
  capabilities: [appointment booking, customer lookup]
  known_limitations: [does not handle rescheduling]   # Agentspect won't test out-of-scope features

world_state:
  entities:
    - id: customer_1
      phone: "+33612345678"
      verified: true
  catalog:
    available_slots: ["2026-05-20T10:00", "2026-05-20T14:00"]
  constraints:
    - "only verified entities can book"
  context:
    current_time: "2026-05-16T08:00"
    language: "fr"

test_config:
  total: 20            # must equal positive + negative, and be >= 1
  positive: 14
  negative: 6

llm_config:
  provider: anthropic  # anthropic | openai | google | openrouter | ollama
  model: claude-sonnet-4-20250514
  # API keys come from the environment, never from this file.
```

Validation is strict and fails fast with a clear message: `total` must equal
`positive + negative`, `total >= 1`, and the reserved `rag` key must not be present
(deferred — see the roadmap).

---

## Assets

Don't hand-write `world_state.entities` for a data-heavy agent — point Agentspect at a file and
let it build them. Add an optional `assets:` block; an `AssetIngestionAgent` reads only a
**bounded, spread-out sample** of the file (never the whole thing, so large files never
reach a prompt) and turns it into world_state material before scenarios are generated.

```yaml
assets:
  - id: customers
    path: ./customers.csv     # relative to base_dir (default: cwd)
    format: csv               # csv | jsonl — inferred from the extension if omitted
    role: entities            # entities | data_sample | knowledge_base
    sample_size: 50           # how many rows to sample across the whole file
    seed: 0                   # deterministic, reproducible sampling
```

```python
report = asyncio.run(run_suite(open("config.yaml").read(), base_dir="./data"))
```

- **`role: entities`** — an LLM maps the sampled rows into `world_state.entities`
  (inferring an id, choosing attributes, keeping variety). Appended to any inline entities.
- **`role: data_sample`** — the sampled rows are placed as-is (deterministic) under
  `catalog/<id>`, for agents that reason over a dataset. Use `target:` (a JSON pointer) to
  place either role elsewhere in world_state.
- **`role: knowledge_base`** — reserved for RAG; not implemented yet.

Sources are pluggable: `csv` and `jsonl` are built in; register your own `AssetLoader`
(Parquet, a Databricks query) with `@register_loader` and it slots into the same pipeline.

---

## The report

`run_suite` returns a dict shaped roughly like this:

```python
{
  "agent_name": "CRM Booking Assistant",
  "total_scenarios": 20,
  "verdict_counts": {"success": 9, "success_unverified": 4, "failure": 2, "suspect": 1, ...},
  "metrics": {
    "task_completion":   {"rate": 0.86, "successful": 12, "total": 14},
    "boundary_adherence":{"rate": 0.83, "successful": 5,  "total": 6},
    "latency":           {"turns": 63, "avg_ms": 812, "p95_ms": 1900, ...},
    # ... one entry per registered metric
  },
  "scenarios": [ {"id": ..., "verdict": ..., "turns": [...], "assertion_results": [...]}, ... ],
  "probe_chains": [ {"parent_id": ..., "probe_id": ..., "probe_type": "probe", ...} ],
  "failure_analysis": {"summary": ..., "constraint_violations": [...], "recommendations": [...]}
}
```

---

## Public API

```python
from agentspect import (
    run_suite,           # async: (yaml_str, progress_callback=None, agent=None) -> report dict
    OrchestratorAgent,   # the LangGraph pipeline, for finer control
    parse_and_validate,  # yaml_str -> (validated model, content hash)
    create_llm_client,   # provider/model -> unified LLMClient
    CallableAdapter,     # wrap a Python callable as the agent under test
    Metric, register, MetricEngine, compute_metrics,   # the metrics system
    # domain models: WorldState, Scenario, Verdict, Assertion, Transcript, Turn, ...
)
```

---

## Configuration

Provider and model are chosen per run via `llm_config`. All five providers sit behind one
`LLMClient` interface; API keys are read from the environment (or a local `.env`), never
from the YAML:

| Provider | Env var |
|----------|---------|
| `anthropic` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `google` | `GOOGLE_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `ollama` | `OLLAMA_BASE_URL` (local, OpenAI-compatible) |

---

## Development

```bash
git clone https://github.com/Bissmella/agentspect
cd agentspect
uv sync --extra dev
uv run pytest            # unit tests mock the LLM — no API key needed
uv run ruff check .
```

---

## Project layout

```
agentspect/
  models/      domain models — world_state, suite, transcript, yaml_input
  agents/      the six Agentspect agents + the LangGraph orchestrator + graph state
  adapters/    HTTP, WebSocket, and in-process callable adapters (same Transcript out of each)
  assets/      pluggable file loaders (CSV/JSONL) + bounded sampling for world_state
  llm/         common LLM interface (Anthropic / OpenAI / Google / OpenRouter / Ollama)
  services/    yaml parsing, placeholder resolution, execution DAG
  metrics/     pluggable metrics — base class, registry, engine, built-ins
examples/      ready-to-run YAML inputs
tests/         unit tests (LLM mocked)
```

---

## Roadmap

- [x] Assets: sample `world_state` entities / data from external files (CSV/JSONL) —
      only a bounded sample is read, so large files never bloat the prompt (see [Assets](#assets))
- [ ] Assets: remote sources (a Databricks query, Parquet) as additional loaders
- [ ] RAG support (the reserved `rag` key / `knowledge_base` asset role): a document
      corpus as ground truth for generating and grading grounded/out-of-scope questions
- [ ] Publish to PyPI (`pip install agentspect`)
- [ ] `agentspect` CLI (`agentspect run config.yaml`) with CI-friendly exit codes
- [ ] Standalone HTML report renderer for local runs
- [ ] Scenario snapshot + LLM record/replay for reproducible, low-cost CI runs
- [ ] pytest plugin and GitHub Action

---

## What Agentspect is not

- Not a metric library and not an observability tool — no instrumentation, no SDK wrappers.
- Not a load-testing tool — concurrency is for running independent suites, not hammering one agent.
- Not a single prompt/response evaluator — it tests behavior over whole conversations.

---

## Contributing

Issues and pull requests are welcome. Please run `uv run pytest` and `uv run ruff check .`
before opening a PR, and add tests for new behavior. New protocol adapters and new metrics
are designed to be added without touching the core — subclass the relevant base and
register it.

---

## License

[Apache-2.0](LICENSE).
