# Voice channel — implementation plan

> Working plan for adding voice-agent support to the OSS `ata` library.
> Scope of this plan: the **thin** foundation (STT/TTS I/O + telemetry capture),
> engineered so the **thick** layer (voice-behaviour metrics) bolts on with no
> rewrite. Thick metrics, adversarial audio, and real telephony are explicitly
> out of scope here and listed under "Later".

---

## Status (branch `feat/voice-channel`)

- ✅ **Phase 0** — `VoiceMeta` on `Turn`, `opening_utterance`/`opening_voice` on `Transcript` (additive).
- ✅ **Phase 1** — `ata/voice/` package: `STTClient`/`TTSClient` + `VoiceIO`, `register_stt`/`register_tts` registry, OpenAI reference provider. Pipecat bridge is **scaffolded** (guarded lazy import; batch↔frame glue is a follow-up).
- ✅ **Phase 2** — `VoiceWebSocketAdapter`: greeting capture, TTS→send→collect→STT round-trip, `end_of_speech`/timeout endpointing, `VoiceMeta` telemetry.
- ✅ **Phase 3** — YAML `voice` block + `voice_websocket` protocol, `create_adapter` routing, orchestrator wiring, `ata[pipecat]` extra, `examples/voice_agent.yaml` + `examples/voice_echo_agent.py` toy target.
- ✅ **Phase 4** — `TimeToFirstAudioMetric` proof metric + `tests/unit/test_voice.py` (13 tests, full suite 243 green).
- ⏭️ **Follow-ups:** finish the Pipecat bridge (batch↔frame + provider auto-construction); the three open questions below; then the thick metrics.

---

## Guiding decisions (settled in discussion)

1. **Black-box, unchanged premise.** The agent under test may be a chained
   pipeline (STT→LLM→TTS) or a speech-native model (Moshi / realtime). ATA neither
   knows nor cares — audio in, audio out.
2. **Thin first.** Thin = STT (agent audio → text) + TTS (user text → audio) around
   the *existing text core*. The UserSimulator, ScenarioGenerator, Scorer,
   world_state and verdict flow stay text and untouched.
3. **Thin must capture voice telemetry.** A metric can only read what's on the
   `Turn`; audio is ephemeral and non-retrofittable. So the thin adapter records
   timing/audio metadata onto each `Turn` even though it computes nothing from it
   yet. This is the bridge that keeps "thick = just add metrics" true.
4. **Thick = pluggable metrics.** Proven by `LatencyMetric` — a hook-based metric
   reading `turn.latency_ms`. Voice metrics are new `@register`ed `Metric`
   subclasses; **zero engine change**.
5. **WS-audio target first.** Self-contained, no telephony account, runnable OSS
   demo. PSTN/Twilio is a later pluggable adapter on the same base.
6. **Deterministic tester.** No neural speech-to-speech simulator — a testing tool
   wants controllable, repeatable timing, not natural conversation.
7. **Don't own the provider matrix.** STT/TTS has dozens of providers and the list
   churns. Minimal `VoiceClient` contract + registry + one reference (OpenAI);
   reach the long tail via an **optional Pipecat bridge**, not by re-wrapping dozens
   of providers. Pipecat chosen over LiveKit Agents for popularity/fit.

---

## What changes vs. what stays

| Component | Change |
|-----------|--------|
| `models/transcript.py` | **additive**: optional `VoiceMeta` on `Turn`; optional opening-utterance capture on `Transcript` |
| `adapters/base.py` | small extension for capturing an agent-first greeting |
| `adapters/` | **new** `voice_ws_adapter.py` |
| `voice/` (new pkg) | **new** `VoiceClient` protocol + registry + OpenAI reference + optional Pipecat bridge |
| `models/yaml_input.py` | **new** optional `voice` config block + validation |
| `config.py` | STT/TTS API keys via env |
| `adapters/ws_adapter.py::create_adapter` | register `voice_websocket` protocol |
| `metrics/` | one proof metric now; the rest later |
| UserSimulator / Scorer / ScenarioGenerator / world_state / verdict flow | **untouched** |

---

## Model changes (Phase 0 — additive, safe first step)

`Turn` gains one optional nested field; text adapters never set it, so existing
metrics and the text path are unaffected:

```python
class VoiceMeta(BaseModel):          # all fields optional
    time_to_first_audio_ms: int | None = None   # start-of-agent-speech, not round-trip
    agent_speech_ms: int | None = None
    user_speech_ms: int | None = None
    silence_gaps_ms: list[int] = []
    interrupted: bool | None = None              # placeholder for thick barge-in
    dtmf: str | None = None
    stt_confidence: float | None = None
    voice: str | None = None                     # which TTS voice ATA spoke with
    accent: str | None = None
    language: str | None = None
    audio_ref: str | None = None                 # blob key for recorded audio

class Turn(BaseModel):
    ...
    voice: VoiceMeta | None = None
```

**Agent-greets-first.** Phone/voice agents speak before the user. Add optional
capture that keeps turn semantics clean (so `ConversationEfficiency`'s
`len(turns)` isn't skewed):

```python
class Transcript(BaseModel):
    ...
    opening_utterance: str | None = None
    opening_voice: VoiceMeta | None = None
```

The UserSimulator receives `opening_utterance` as context for its first turn; the
Scorer sees it alongside the transcript. (Decision to confirm: opening as a field
vs. a synthetic user_message="" turn — field chosen to avoid polluting turn counts.)

---

## STT/TTS providers — registry + reference + optional bridge (Phase 1)

ATA does **not** own the STT/TTS provider matrix (dozens of providers, churning
list). Because **thin** needs only batch request/response — `synthesize(text)->
audio`, `transcribe(audio)->text+confidence` — a provider is ~30 lines. So the
strategy is: define one contract, make BYO trivial, ship one reference, and reach
the long tail through Pipecat rather than re-wrapping it.

New package `ata/voice/`:

```
ata/voice/
  __init__.py
  client.py        # VoiceClient protocol: transcribe / synthesize
  registry.py      # @register — mirrors metrics/registry.py & the adapter registry
  providers/
    openai.py      # the one built-in reference (already a dependency)
  bridges/
    pipecat.py     # optional: VoiceClient over Pipecat services (extra dep)
```

- `transcribe(audio_bytes) -> TranscriptionResult(text, confidence, language)`
- `synthesize(text, *, voice, language, accent) -> audio_bytes`
- **Reference provider: OpenAI** — Whisper STT + TTS, no new heavy dep.
- **BYO provider:** subclass `VoiceClient` and `@register` it — the same pattern
  users already have for metrics and adapters.
- **Optional Pipecat bridge** (`pip install ata[pipecat]`, never a core dep): one
  adapter exposing Pipecat's STT/TTS services as a `VoiceClient`, buying dozens of
  providers *maintained upstream* — and Pipecat also supplies VAD/endpointing
  (see Phase 2).
- `voice` / `language` / `accent` are passed through **opaquely** as
  provider-specific strings — ATA does not normalize voice IDs across providers.
- Keys via env (`OPENAI_API_KEY`, etc.), never in YAML — same rule as LLM.

---

## Voice WS adapter (Phase 2 — the core work)

`ata/adapters/voice_ws_adapter.py`, subclass of `ProtocolAdapter`. Keeps the
lockstep `send_turn(session_id, text) -> Turn` contract for thin:

- `start_session`: open the audio WS; capture the agent's **opening greeting**
  (receive audio until endpoint → STT) → store on `Transcript.opening_utterance`.
- `send_turn(text)`:
  1. TTS the user text → audio (record `voice/accent/language` used)
  2. stream audio frames to the agent
  3. receive agent audio frames until **end-of-speech** (see endpointing below),
     stamping `time_to_first_audio_ms` at the first inbound frame
  4. STT the agent audio → `agent_response` text + `stt_confidence`
  5. return `Turn` with `voice=VoiceMeta(...)` populated
- `end_session` / `close`: tear down the connection.
- Non-response / silence-past-timeout → `Turn(error="TIMEOUT")` → verdict `ERROR`
  (consistent with existing adapters — never an exception).

**Endpointing (how we know the agent stopped talking).** The one genuinely new
piece of complexity. Options, in preference order:
1. honour an explicit turn-complete / end-of-speech signal in the WS protocol if
   the target provides one;
2. use Pipecat's VAD (Silero) when the `[pipecat]` extra is installed;
3. otherwise a simple silence endpointer (energy threshold + hangover window).
Configurable via the `voice` block because it's target-dependent.

---

## Wiring (Phase 3)

- `create_adapter` (`ws_adapter.py:103`): add `"voice_websocket"` → `VoiceWSAdapter`.
- `models/yaml_input.py`: optional `voice` block, validated on load:
  ```yaml
  agent_under_test:
    protocol: voice_websocket
  voice:
    stt: { provider: openai, model: ... }
    tts: { provider: openai, voice: alloy, language: fr, accent: null }
    endpointing: { mode: vad, silence_ms: 700 }
  ```
- `config.py`: surface STT/TTS provider keys from env.
- Example `examples/voice_agent.yaml` + a **toy WS-audio target** under
  `examples/` (a tiny greet-then-echo voice bot) so the demo runs end-to-end with
  no telephony account.

---

## Proof metric + tests (Phase 4)

- Ship **one** voice metric now — `time_to_first_audio` (percentiles, same shape as
  `LatencyMetric`) — purely to validate the telemetry path end-to-end and stand as
  the template for the rest. Register it; no engine change.
- Tests: a **mock voice adapter** + fake STT/TTS (deterministic canned audio↔text)
  so the whole path is unit-testable without network or provider keys. Assert:
  `VoiceMeta` is populated, opening utterance captured, TIMEOUT → error Turn, the
  proof metric computes.

---

## Later (thick — out of scope for this plan)

Each is a new `@register`ed metric or an adapter behaviour, built on the telemetry
captured above — no rewrite of Phase 0–4:
- **Voice-behaviour metrics:** barge-in handling, response-latency thresholds,
  silence/timeout recovery, talk-over, DTMF entry.
- **Barge-in orchestration:** deterministic "start user audio N ms into agent
  speech" trigger in the adapter (sets `VoiceMeta.interrupted`).
- **Latency-threshold assertions:** a new assertion type, or reuse `world_state`/
  `transcript` assertions over `VoiceMeta`.
- **Adversarial audio:** ScenarioGenerator varies voice/accent/noise; a metric
  scores comprehension degradation via `stt_confidence` + retries. Anchored on
  `world_state.context.language`.
- **PSTN / telephony adapter:** Twilio/Telnyx/SIP + media bridge + public callback,
  as a pluggable adapter on the same base.

---

## Settled
- Reference provider: **OpenAI** (Whisper STT + TTS; already a dependency).
- Long-tail providers + VAD: **optional Pipecat bridge** (`ata[pipecat]`), not core.

## Open questions to confirm before coding

1. Opening-utterance representation — field (planned) vs. synthetic turn.
2. Endpointing default when `[pipecat]` is absent — ship the energy-based fallback
   in v1, or require the `[pipecat]` extra to get VAD at all?
3. Do we record raw audio to blob storage in the OSS lib, or leave `audio_ref`
   as a hook the platform repo fills? (Lib stays infra-light → likely a hook.)
