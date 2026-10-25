from typing import Any

from pydantic import BaseModel, Field, model_validator


class AgentUnderTest(BaseModel):
    name: str
    url: str | None = None
    protocol: str = Field(pattern=r"^(http|websocket|callable|voice_websocket)$")
    description: str
    capabilities: list[str] = Field(default_factory=list)
    known_limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _require_url_for_network_protocols(self):
        if self.protocol in ("http", "websocket", "voice_websocket") and not self.url:
            raise ValueError(f"url is required for protocol '{self.protocol}'")
        return self


class WorldStateInput(BaseModel):
    entities: list[dict[str, Any]] = Field(default_factory=list)
    catalog: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class TestConfig(BaseModel):
    total: int = Field(ge=1)
    positive: int = Field(ge=0)
    negative: int = Field(ge=0)


class LLMConfig(BaseModel):
    provider: str = Field(pattern=r"^(anthropic|openai|google|openrouter|groq|ollama)$")
    model: str


class STTSpec(BaseModel):
    provider: str = "openai"
    model: str | None = None


class TTSSpec(BaseModel):
    provider: str = "openai"
    model: str | None = None
    voice: str | None = None
    language: str | None = None
    accent: str | None = None


class EndpointingSpec(BaseModel):
    """How ATA decides a speaker finished — like a real call, not a control frame.

    mode: 'vad' (default) ends a turn on the speech→silence edge (VAD on a
    continuous stream, or an inter-frame silence gap on a frame protocol); neither
    side sends or needs an end_of_speech signal. 'signal' is a legacy opt-in for
    self-framing WS agents that do emit end_of_speech.
    silence_ms: how long of quiet marks the end of a turn.
    """

    mode: str = Field(default="vad", pattern=r"^(signal|vad)$")
    silence_ms: int = Field(default=700, ge=0)


class VoiceConfig(BaseModel):
    stt: STTSpec = Field(default_factory=STTSpec)
    tts: TTSSpec = Field(default_factory=TTSSpec)
    endpointing: EndpointingSpec = Field(default_factory=EndpointingSpec)


class AssetSpec(BaseModel):
    """An external data source ATA samples from to build test material.

    ``format`` is inferred from the path extension when omitted. ``role`` decides
    what the sample becomes: ``entities`` / ``data_sample`` are ingested into
    world_state; ``knowledge_base`` is reserved for RAG. ``target`` is an optional
    JSON pointer overriding where the ingested data is placed in world_state.
    """

    id: str
    path: str
    format: str | None = Field(default=None, pattern=r"^(csv|jsonl)$")
    role: str = Field(default="entities", pattern=r"^(entities|data_sample|knowledge_base)$")
    sample_size: int = Field(default=50, ge=1)
    seed: int = 0
    description: str | None = None
    target: str | None = None


class YAMLInput(BaseModel):
    agent_under_test: AgentUnderTest
    world_state: WorldStateInput
    test_config: TestConfig
    llm_config: LLMConfig
    assets: list[AssetSpec] = Field(default_factory=list)
    voice: VoiceConfig | None = None

    @model_validator(mode="after")
    def _require_voice_for_voice_protocol(self):
        if self.agent_under_test.protocol == "voice_websocket" and self.voice is None:
            # A default voice config (OpenAI STT/TTS) is enough to run.
            self.voice = VoiceConfig()
        return self
