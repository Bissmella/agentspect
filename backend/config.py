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

    model_config = {"env_file": ".env", "extra": "ignore"}

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
