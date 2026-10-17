from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables or a local .env file.

    DATABASE_URL has no default on purpose: credentials must never be hard-coded.
    The app refuses to start until it is configured.
    """

    app_name: str = "ITSM AI Agent"
    database_url: str

    # --- LLM (used for ticket classification) ---
    # No model name or key is baked into the code. If they are missing, classification
    # fails safely (the ticket goes to awaiting_human); it never invents a result.
    llm_provider: Literal["anthropic"] = "anthropic"
    llm_model: str | None = None
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = 30.0  # per request; one place to change it
    llm_max_retries: int = 1  # the SDK retries transient errors (429/5xx/connection) this often

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
