from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings

_DOCKER_HOST_MAP = {
    "localhost:5432": "postgres:5432",
    "localhost:6379": "redis:6379",
    "localhost:9000": "minio:9000",
    "localhost:11434": "host.docker.internal:11434",
}

_DOCKER_FIELDS = ("database_url", "redis_url", "s3_endpoint_url", "ollama_base_url")

# Voice STT/TTS provider -> (Settings attr, env var name). ollama/custom need no key.
_PROVIDER_KEYS = {
    "openai": ("openai_api_key", "OPENAI_API_KEY"),
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY"),
    "google": ("google_api_key", "GOOGLE_API_KEY"),
    "openrouter": ("openrouter_api_key", "OPENROUTER_API_KEY"),
}


def _is_docker() -> bool:
    return Path("/.dockerenv").exists()


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://ata:ata_dev@localhost:5432/ata"
    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint_url: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket_name: str = "ata-blobs"

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"

    # Celery per-suite time limits (seconds). Voice suites stream real-time audio
    # and legitimately run for minutes, so they get a longer leash than text.
    task_soft_time_limit: int = 1800         # text suites (prev. unbounded; generous to avoid reaping multi-scenario runs)
    voice_task_soft_time_limit: int = 2400   # voice suites
    task_time_limit_grace: int = 60          # hard limit = soft + grace

    model_config = {"env_file": ".env", "extra": "ignore"}

    def task_time_limits(self, protocol: str | None) -> tuple[int, int]:
        """(soft, hard) Celery limits in seconds for a suite's protocol."""
        soft = (
            self.voice_task_soft_time_limit
            if protocol == "voice_websocket"
            else self.task_soft_time_limit
        )
        return soft, soft + self.task_time_limit_grace

    def missing_voice_keys(self, *providers: str) -> list[str]:
        """Env var names required by the given STT/TTS providers that are unset.

        Unknown (custom/pipecat) providers can't be validated here, so they pass.
        """
        missing = []
        for prov in dict.fromkeys(providers):  # dedupe, keep order
            entry = _PROVIDER_KEYS.get(prov)
            if entry and not getattr(self, entry[0]):
                missing.append(entry[1])
        return missing

    @model_validator(mode="after")
    def _resolve_docker_urls(self):
        if not _is_docker():
            return self
        for field in _DOCKER_FIELDS:
            value = getattr(self, field)
            for old, new in _DOCKER_HOST_MAP.items():
                value = value.replace(old, new)
            object.__setattr__(self, field, value)
        return self


settings = Settings()
