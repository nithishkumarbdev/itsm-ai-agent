from functools import lru_cache

from fastapi import Depends

from app.core.config import settings
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMClient
from app.services.classification import ClassificationService


@lru_cache
def _real_llm_client() -> AnthropicClient:
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else None
    return AnthropicClient(
        api_key=api_key or None,
        model=settings.llm_model or None,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def get_llm_client() -> LLMClient:
    """The configured LLM. Tests replace this dependency with a mock (see tests/conftest.py)."""
    return _real_llm_client()


def get_classification_service(llm: LLMClient = Depends(get_llm_client)) -> ClassificationService:
    return ClassificationService(llm)
