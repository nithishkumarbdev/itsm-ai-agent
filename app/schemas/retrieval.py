import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.classification import Category


class KnowledgeSearchRequest(BaseModel):
    """A search of the knowledge base. Meant for debugging and tuning retrieval."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [{"query": "VPN disconnects every few minutes", "top_k": 5, "category": "vpn"}]
        },
    )

    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20, description="Defaults to RETRIEVAL_TOP_K.")
    min_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum similarity. Defaults to RETRIEVAL_MIN_SCORE."
    )
    category: Category | None = Field(
        default=None, description="Ranks documents of this category first. Other categories can still match."
    )

    @field_validator("query")
    @classmethod
    def query_must_be_meaningful(cls, value: str) -> str:
        if not any(ch.isalnum() for ch in value):
            raise ValueError("must contain at least one letter or digit")
        return value


class RetrievedChunkRead(BaseModel):
    """One retrieved piece of knowledge and where it came from."""

    model_config = ConfigDict(from_attributes=True)

    document_id: uuid.UUID
    document_title: str
    category: str
    version: int
    chunk_id: uuid.UUID
    chunk_index: int
    section: str
    content: str
    similarity_score: float = Field(description="Cosine similarity between the query and the chunk.")
    category_match: bool


class KnowledgeSearchResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    results: list[RetrievedChunkRead] = Field(description="Empty means no sufficiently relevant knowledge.")
    candidates_considered: int = Field(description="Chunks that were eligible. 0 means nothing is indexed.")
    top_k: int
    min_score: float


class IndexResultRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    document_id: uuid.UUID
    title: str
    status: Literal["indexed", "unchanged", "skipped_inactive"]
    chunk_count: int
    embedding_model: str
