from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables or a local .env file.

    DATABASE_URL has no default on purpose: credentials must never be hard-coded.
    The app refuses to start until it is configured.
    """

    app_name: str = "ITSM AI Agent"
    database_url: str

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
