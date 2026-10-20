from typing import Literal

from pydantic import Field, SecretStr
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

    # --- Embeddings and retrieval (knowledge search) ---
    # The real provider is any service with an OpenAI-style POST /embeddings endpoint.
    # No key or model name is baked into the code. If they are missing, indexing and
    # retrieval fail safely; they never return made-up results.
    embedding_provider: Literal["openai"] = "openai"
    embedding_model: str | None = None
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_dimension: int = Field(default=1536, gt=0)  # must match the model's output size
    embedding_timeout_seconds: float = 30.0

    retrieval_top_k: int = Field(default=5, ge=1, le=20)  # most chunks returned per search
    # Minimum cosine similarity for a chunk to count as relevant. This is a starting
    # point for text-embedding-3-small, NOT a validated value: tune it with real data.
    retrieval_min_score: float = Field(default=0.30, ge=0.0, le=1.0)

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
