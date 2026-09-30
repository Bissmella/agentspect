# Thick voice channel — implementation plan

> Companion to `voice-plan.md` (the thin foundation). Thick = voice-behaviour
> testing that thin's lockstep model can't express: barge-in, talk-over, silence
> recovery, DTMF, and precise turn-taking timing.

---

## Status (branch `voice_channel_metrics`)

- ✅ **Stage A** — streaming capture (`_AudioCapture` timestamps → `agent_speech_ms`,
  `silence_gaps_ms`) + metrics `dead_air`, `agent_speech_duration`. No model change
  (reserved `VoiceMeta` fields), no Pipecat.
  - `turn_taking_latency` was **not** added as a separate metric: in the single-blob
    thin send it is identical to the existing `time_to_first_audio` metric. A
    genuinely precise turn-taking gap (VAD both sides) is Stage B.
- ⏭️ **Stage B** — see below. Recommended order: **B3 schema first** (additive,
  Pipecat-independent), then **verify Pipecat's VAD/interruption API**, then B2/B1
  (event model + runtime designed against that API), then B4.

---

## Core realization

For thin, "thick = just add metrics" held because thin already captured the
telemetry. Thick breaks that. The wall is the adapter contract:

`send_turn(text) -> Turn` is **lockstep** — speak a whole turn, then listen for a
whole turn. Every thick behaviour is **duplex and time-overlapping** (speak while
the agent speaks, detect simultaneous speech, act at a precise moment). So thick
needs a **new capture model first** (streaming, timestamped, duplex); the metrics
read that. Same lesson as Phase 0, one layer down. The reserved `VoiceMeta` fields
(`interrupted`, `silence_gaps_ms`, `agent_speech_ms`, `dtmf`) anticipated this.

## Settled decisions

- **Two stages, A before B.** A front-loads the tractable metrics and builds the
  timestamp foundation; B isolates the hard duplex work.
- **Substrate: Pipecat for Stage B.** Don't reinvent duplex I/O + VAD +
  interruption — Pipecat's frame pipeline provides them, and it's already the
  STT/TTS provider path (`ata[pipecat]`). Consequence: **thick is gated on the
  `ata[pipecat]` extra; thin stays fully functional without it.**
- **Deterministic tester.** Scripted, timed adversarial actions — not a neural
  speech-to-speech simulator. Timing is "≈T ± tolerance" (real audio has jitter).
- **Transport-neutral.** Capture and metrics stay protocol-agnostic so the
  platform's telephony adapter reuses them.

---

## Stage A — streaming capture + observational metrics

No injection, no duplex, no scenario/generator changes. Just a richer read of the
agent's audio stream. Low risk, immediate value, and it lays the timestamp
foundation Stage B extends.

**Capture.** Extend the existing frame-collection loop (`voice_ws_adapter.
_collect_agent_audio` already iterates frames) to timestamp frame arrivals and
populate, per turn:
- `agent_speech_ms` — first-to-last audio-frame span
- `silence_gaps_ms` — inter-frame gaps above a threshold (dead air mid-response)
- precise turn-taking latency — last user-audio-sent → first agent audio (finer
  than the round-trip `latency_ms`; `time_to_first_audio_ms` already approximates)

No Pipecat needed here — it's bookkeeping over frames we already receive. Keep the
loop structured so Stage B can hang injection events off the same timeline.

**Metrics** (`@register`ed, read `VoiceMeta`) — shipped:
- `dead_air` — count / max / total of mid-response silence gaps
- `agent_speech_duration` — how long the agent talks per turn
- turn-taking latency: served by the existing `time_to_first_audio` metric (see Status).

**Touches:** `voice_ws_adapter.py` (timestamped collect), `metrics/builtin.py`, tests.

---

## Stage B — duplex + scripted adversarial actions (the hard part)

Needs ATA to act at a precise moment relative to the agent's audio. Four sub-parts,
each with a real design question.

### B1. Duplex session model (on Pipecat)
A new execution path *alongside* `send_turn` — the lockstep contract can't do
overlap. A `DuplexVoiceSession` drives a Pipecat pipeline: plays user audio frames,
receives agent frames, consumes Pipecat's VAD/interruption events for speech
boundaries, and can inject user audio mid-agent-speech.
- Open: exact Pipecat API for VAD events + interruption — **verify at build time**,
  don't assume.
- The orchestrator/user-simulator needs a duplex run branch for voice scenarios
  that carry timed actions.

### B2. Event timeline
Injection events don't exist without injection, so this is genuinely new capture.
Add a `voice_events` list (e.g. `{t_ms, kind, party}`, kind ∈ `speech_start`,
`speech_end`, `silence`, `barge_in_injected`, `dtmf_sent`, `agent_yielded`).
Metrics compute over the timeline.

### B3. Scenario schema + generator
Today a turn is a plain string. Timed actions need a richer form, e.g.:
```yaml
turns:
  - text: "I'd like to book a slot"
  - text: "actually, wait—"
    barge_in_at_ms: 500        # start speaking 500ms into the agent's response
```
Plus: ScenarioGenerator (LLM) emits these for robustness scenarios, and a new
assertion kind (`voice_behavior`: `yields_on_barge_in`, `reprompts_after_silence`,
`accepts_dtmf`). **This is the largest surface of B** — schema, generator prompt,
scorer.

### B4. Metrics + determinism
- Metrics: `barge_in_response` (fraction of injected barge-ins where the agent
  yielded within tolerance), `talk_over_rate` (agent speaks while user is
  speaking), `silence_recovery` (agent re-prompts after injected user silence),
  `dtmf_accepted`.
- Determinism: injected **simulated clock** into the session; **fake duplex
  transport** for tests; assertions use **timing tolerances**, not exact ms.

---

## Risks / open questions (Stage B)
- Pipecat's interruption/VAD API surface — verify before committing B1.
- Timing-dependent tests are flaky by nature → the fake transport + simulated clock
  are not optional, they're the only way to test B reliably.
- Scenario-schema change ripples into the generator and scorer — scope it as its
  own sub-milestone.
- Barge-in realism vs determinism: "500 ms" is aspirational over a real network;
  decide the tolerance contract early.
