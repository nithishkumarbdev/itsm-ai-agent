from functools import lru_cache

from fastapi import Depends

from app.core.config import settings
from app.embeddings.base import EmbeddingClient
from app.embeddings.openai_compatible import OpenAICompatibleEmbeddings
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMClient
from app.services.classification import ClassificationService
from app.services.retrieval import RetrievalService


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


@lru_cache
def _real_embedding_client() -> OpenAICompatibleEmbeddings:
    api_key = settings.embedding_api_key.get_secret_value() if settings.embedding_api_key else None
    return OpenAICompatibleEmbeddings(
        api_key=api_key or None,
        model=settings.embedding_model or None,
        dimension=settings.embedding_dimension,
        base_url=settings.embedding_base_url,
        timeout=settings.embedding_timeout_seconds,
    )


def get_embedding_client() -> EmbeddingClient:
    """The configured embedding model. Tests replace this dependency with a mock."""
    return _real_embedding_client()


def get_retrieval_service(embedder: EmbeddingClient = Depends(get_embedding_client)) -> RetrievalService:
    return RetrievalService(embedder, top_k=settings.retrieval_top_k, min_score=settings.retrieval_min_score)
