# Capability Map: Voice Channel

**Status:** approved 2026-10-11 · **Initiative:** add a voice channel (telephony) to the ATA platform.

## Context

The engine (`agentspect`) already implements real-time voice: a `voice_websocket`
protocol with a duplex adapter (streaming TTS out, audio capture + VAD + streaming
STT in), OpenAI STT/TTS provider, a pipecat bridge for "thick" voice, and per-turn
`VoiceMeta` (TTFA, speech/silence spans, DTMF, stt_confidence) + a `VoiceEvent`
timeline. The data model has an `audio_ref` slot for recorded audio. The platform,
by contrast, has **no** voice code: it stores only text transcripts and the UI is
text-only.

The engine's duplex loop is transport-agnostic — it drives a **duck-typed
`connection`** (`.send(json)/.recv()->json/.close()`) speaking a tiny JSON audio
frame protocol. This lets the platform own telephony end to end by supplying its own
Telnyx-backed `connection`, with the engine unchanged except for one small seam.

## Decisions (locked)

- **Telephony provider:** Telnyx (Media Streaming over WebSocket; pipecat ships a Telnyx serializer).
- **Call direction:** outbound — ATA (the test "user") places calls to the agent-under-test's number.
- **Recording:** simplest first — one mixed WAV per session, same S3 retention as transcripts.
- **Engine changes:** land as RFCs on the OSS `agentspect` repo + a git-tag bump. Never patch/fork agentspect locally.
- **STT/TTS stays in the engine** (VoiceIO). The platform only transcodes WAV ⇄ G.711 μ-law, never speech.

## Modules

| Module id | Responsibility | Side | Depends on |
|---|---|---|---|
| `voice-run` | Surface `voice_websocket` + `VoiceConfig` through platform YAML validation, Celery task, DB, and progress events so a voice suite runs end to end and stores voice telemetry | platform | — |
| `telephony` | Telnyx outbound origination, public media-stream WS endpoint, μ-law⇄WAV transcode, a `TelnyxConnection` shim implementing the engine's send/recv protocol, call lifecycle + credential/number config | platform + engine RFC #1 | `voice-run` |
| `recording` | Tee telephony media → WAV → S3 → populate `audio_ref` + DB keys; extend existing text-transcript storage | platform | `telephony` |
| `voice-ui` | Audio playback in transcript/report, voice-metric panels (TTFA/latency/silence), voice+telephony config in NewRun, live call/turn indicators | frontend | `voice-run`, `recording`, `telephony` |

**Dependency direction (no cycles):** `voice-run → {telephony} → recording → voice-ui`
**Build order:** `voice-run` → `telephony` → `recording` → `voice-ui` (UI metric views may begin right after `voice-run`).

## agentspect RFCs (engine-side, git-tag bump — no local patch)

1. **Required, small — injectable-connection voice adapter.** Expose a way to run the
   voice duplex loop over a host-supplied `connection`, mirroring `CallableAdapter`
   for HTTP. Internals already support it (`_connections` + duck-typed send/recv); the
   RFC is "expose it," not "build it." Unblocks `telephony`.
2. **Deferred/optional — host audio-capture hook.** Surface the audio the voice adapter
   already captures so a host can persist it. Only needed to record *direct*
   `voice_websocket` runs; telephony recording is platform-side (the media stream
   carries both legs), so this is not on the telephony path.

## Specs

Each module gets its own spec alongside this map, named by id: `SPEC-voice-run.md`,
`SPEC-telephony.md`, `SPEC-recording.md`, `SPEC-voice-ui.md`. Specs are written and
approved one at a time in build order.
