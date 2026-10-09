from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for the ATA engine.

    API keys are read from the environment (or a local .env file), never from
    the input YAML. Provider and model are chosen per-run via ``llm_config``.
    """

    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/v1"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
