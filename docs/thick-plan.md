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
- ✅ **Stage B3 schema** — `VoiceAction`/`VoiceActionType` + `Scenario.voice_actions`
  (additive, validated). Generator + `voice_behavior` assertion still to do.
- ✅ **Pipecat VAD API verified** — `SileroVADAnalyzer.analyze_audio(pcm) -> VADState`,
  standalone, `pipecat-ai[silero]`, local CPU. Only the VAD analyzer is used (ATA is
  the caller, not a bot). Extra updated to `pipecat-ai[silero]`.
- ✅ **B2 event timeline** — `VoiceEvent`/`VoiceEventKind`, carried **per-turn** on
  `Turn.voice_events`; `Transcript.voice_events` is a derived property (flattens
  turns). So the timeline flows through `send_turn`'s return — no extra adapter
  method.
- ✅ **B1 reworked into a concurrent streaming mixin (issues #8–11).** Standalone
  `DuplexVoiceSession` deleted. `VoiceDuplexMixin` (`ata/adapters/voice_duplex_mixin.py`),
  mixed into `voice_ws_adapter`, exposes **only `send_turn`** and runs **two
  concurrent activities**: outbound (token stream → `voice_io.synthesize_stream` →
  `asyncio.Queue` → frames sent as produced; `ATA_SPEECH_START` at first frame sent)
  and inbound (a parallel recv task doing VAD + TTFA/gap/speech-event capture).
  `send_turn` takes `str` or `AsyncIterator[str]`. **VAD optional** (guarded
  `SileroVadDetector`); thin voice runs with no pipecat.
  - **Token streaming wired**: `LLMClient.chat_stream` (real streaming for Anthropic
    + OpenAI; one-chunk default for Google/compatible) and the simulator pipes the
    token stream straight into `send_turn` for streaming-capable (voice) adapters on
    adapted turns — verified end-to-end with a fake streaming LLM.
  - **Streaming both directions wired**: outbound token→`synthesize_stream`→frames,
    and inbound runs **continuous STT** — a third concurrent activity consumes audio
    chunks off an `asyncio.Queue` via `VoiceIO.transcribe_stream`, assembling the
    agent utterance as audio arrives (not one batch call at the end).
  - Only the *codec implementations* stay batch-backed behind these interfaces
    (`synthesize_stream` word-buffers via batch TTS; `transcribe_stream` default
    transcribes once). The real **Pipecat streaming TTS/STT** drop in behind
    `VoiceIO` — deferred to an env with `ata[pipecat]` + provider keys, since
    Pipecat's STT/TTS services need pipeline `setup()` and can't be built/validated
    in this repo. Tested here with batch + fake-streaming `VoiceIO`.
- ⏭️ **Deferred, by decision:**
  - **Interruption/barge-in** — execution + metric, and the B3 `voice_actions` schema
    reworked from turn-indexed `at_ms` to a **scenario-level interruption hint**
    (`voice_actions` is parked, unused, until then).
  - **Real streaming** — per-provider `LLMClient` token streaming, Pipecat streaming
    STT/TTS, and the simulator streaming tokens into `send_turn`. The shape is in
    place (token-stream input, optional VAD); content TTS/STT is still batch under
    the hood, so latency isn't reduced yet — these drop in behind `VoiceIO`.
- ⏭️ **Next (not deferred):** **B4 observational metrics** over the per-turn events +
  `VoiceMeta` (dead-air and agent-speech already shipped; add VAD-based ones), and
  wiring the voice adapter's VAD into `create_adapter` when `ata[pipecat]` is present.

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
- **Substrate: Pipecat for Stage B — but only its VAD analyzer** (verified, see
  Status). ATA is the *caller*, not a bot, so Pipecat's pipeline/transport/
  interruption machinery is irrelevant; the single piece ATA needs is
  `SileroVADAnalyzer` (standalone `await analyze_audio(pcm) -> VADState`, local CPU).
  Duplex orchestration, barge-in injection timing, and the transport stay ATA's own
  (over the existing WS); content STT/TTS stays thin's batch `VoiceIO`. Consequence:
  **thick is gated on the `ata[pipecat]` extra (= `pipecat-ai[silero]`); it is one
  guarded import of one class; thin stays fully functional without it.**
- **Deterministic tester.** Scripted, timed adversarial actions — not a neural
  speech-to-speech simulator. Timing is "≈T ± tolerance" (real audio has jitter).
- **Transport-neutral.** Capture and metrics stay protocol-agnostic so the
  platform's telephony adapter reuses them.
- **Audio is just a channel — the text core stays primary (guardrail).** ATA must
  never become audio-first. The text core (models, scorer, verdicts, metrics
  engine, user-simulator loop) is transport-agnostic; voice lives entirely behind
  the `ProtocolAdapter` seam with `VoiceMeta`/events as optional fields. Thick's
  duplex path (B1) is an **opt-in branch taken only for voice scenarios carrying
  `voice_actions`** — normal text agents and plain thin-voice keep the lockstep
  `send_turn` trunk untouched.
- **OpenAI Realtime API (ORA) framing — deferred, optional adapter track.** Unmute,
  OpenAI Realtime, and others speak a dialect of ORA over WS; adopting it in a
  voice adapter would reach real agents directly. But it is only framing *inside*
  an adapter (invisible to the core), orthogonal to thick, and B1 works against a
  transport abstraction regardless — so it is logged as a separate future track,
  not on the thick path.

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

### B1. Duplex session model (ATA's own, VAD from Pipecat)
A new execution path *alongside* `send_turn` — the lockstep contract can't do
overlap. A `DuplexVoiceSession` runs two concurrent asyncio tasks over the existing
WS: one streams ATA's (batch-TTS'd) audio out, one reads the agent's audio in and
feeds it to `SileroVADAnalyzer.analyze_audio` to mark the agent's speech
start/stop. A timed action (e.g. `barge_in_at_ms`) schedules *when* the outbound
task starts — ATA controls its own injection; no Pipecat transport/interruption.
- Audio format: VAD needs 16 kHz mono 16-bit PCM in 512-sample frames → ATA must
  buffer/resample inbound audio to that. Target audio format is a real integration
  detail (flag in risks).
- The orchestrator/user-simulator needs a duplex run branch for voice scenarios
  that carry `voice_actions`.

### B2. Event timeline
Injection events don't exist without injection, so this is genuinely new capture.
Add a `voice_events` list (`{t_ms, kind, party}`, kind ∈ `speech_start`,
`speech_end`, `silence`, `barge_in_injected`, `dtmf_sent`, `agent_yielded`).
Agent `speech_start`/`speech_end` come from `VADState` transitions
(`QUIET→SPEAKING`, `SPEAKING→QUIET`); injection events are stamped by ATA. Metrics
compute over the timeline.

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
- ✅ Pipecat VAD API verified — `SileroVADAnalyzer.analyze_audio(pcm) -> VADState`,
  standalone, `pipecat-ai[silero]`, local CPU. (The pipeline/interruption layer is
  not used.)
- **Audio format** is the new real integration detail: VAD needs 16 kHz mono 16-bit
  PCM in 512-sample frames; the agent's inbound audio may be any format → ATA
  buffers/resamples. Decide how ATA learns the target's audio format (config vs.
  sniff).
- Timing-dependent tests are flaky by nature → the fake transport + simulated clock
  (and a fake VAD returning scripted states) are the only way to test B reliably.
- Scenario-schema change ripples into the generator and scorer — B3 schema landed;
  generator + a `voice_behavior` assertion kind remain.
- Barge-in realism vs determinism: "500 ms" is aspirational over a real network;
  decide the tolerance contract early.
