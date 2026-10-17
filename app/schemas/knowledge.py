import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.classification import Category  # same categories as ticket classification

TITLE_MAX = 200
# A document shorter than this cannot be a real procedure; it would only be noise for retrieval.
CONTENT_MIN = 20
CONTENT_MAX = 50_000


def _must_contain_text(value: str | None) -> str | None:
    if value is not None and not any(ch.isalnum() for ch in value):
        raise ValueError("must contain at least one letter or digit")
    return value


class KnowledgeDocumentCreate(BaseModel):
    """A new knowledge document."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "title": "VPN Troubleshooting Guide",
                    "category": "vpn",
                    "content": "## Symptoms\n- The VPN client keeps disconnecting.\n\n## Troubleshooting Steps\n1. Reconnect.",
                    "version": 1,
                    "metadata": {"department": "IT Support", "source": "internal_kb"},
                }
            ]
        },
    )

    title: str = Field(min_length=1, max_length=TITLE_MAX)
    category: Category
    content: str = Field(min_length=CONTENT_MIN, max_length=CONTENT_MAX)
    version: int = Field(default=1, ge=1)
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict, description="Free-form document facts.")

    @field_validator("title", "content")
    @classmethod
    def text_must_be_meaningful(cls, value: str | None) -> str | None:
        return _must_contain_text(value)


class KnowledgeDocumentUpdate(BaseModel):
    """Changes to a document. Send only the fields to change.

    There is no `version` here: the backend raises it when the content changes.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str | None = Field(default=None, min_length=1, max_length=TITLE_MAX)
    category: Category | None = None
    content: str | None = Field(default=None, min_length=CONTENT_MIN, max_length=CONTENT_MAX)
    metadata: dict[str, Any] | None = None
    is_active: bool | None = None

    @field_validator("title", "content")
    @classmethod
    def text_must_be_meaningful(cls, value: str | None) -> str | None:
        return _must_contain_text(value)

    @model_validator(mode="after")
    def at_least_one_real_field(self):
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided")
        nulls = sorted(name for name in self.model_fields_set if getattr(self, name) is None)
        if nulls:
            raise ValueError(f"fields cannot be null: {', '.join(nulls)}")
        return self


class KnowledgeDocumentSummary(BaseModel):
    """A document without its (possibly long) content. Used for listings."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    category: Category
    version: int
    is_active: bool
    # Read from the ORM attribute doc_metadata, exposed to clients as "metadata".
    metadata: dict[str, Any] = Field(validation_alias="doc_metadata")
    created_at: datetime
    updated_at: datetime


class KnowledgeDocumentRead(KnowledgeDocumentSummary):
    """A full document, including its content."""

    content: str
