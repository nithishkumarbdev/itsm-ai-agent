import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.models import TicketStatus

# Channels a ticket can arrive from. To support a new channel, add it here.
TicketSource = Literal["web", "email", "slack", "other"]


class TicketCreate(BaseModel):
    """What a client may send when opening a ticket.

    extra="forbid": a client cannot send status, category, assigned_team, etc.
    Only the backend sets those.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "source": "web",
                    "requester": "user@example.com",
                    "title": "VPN not connecting",
                    "description": "I cannot connect to the company VPN from home.",
                }
            ]
        },
    )

    source: TicketSource = Field(description="Where the ticket came from.")
    requester: EmailStr = Field(description="Email address of the person reporting the issue.")
    title: str = Field(min_length=1, max_length=200, description="Short summary of the issue.")
    description: str = Field(
        min_length=1, max_length=5000, description="Full description of the issue."
    )

    @field_validator("title", "description")
    @classmethod
    def must_contain_text(cls, value: str) -> str:
        # Whitespace is already stripped; this also rejects values like "!!!" or "---".
        if not any(ch.isalnum() for ch in value):
            raise ValueError("must contain at least one letter or digit")
        return value


class TicketRead(BaseModel):
    """A ticket as stored in the database and returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    source: str
    requester: str
    title: str
    description: str
    status: TicketStatus = Field(description="Lifecycle status. New tickets start as 'new'.")

    # The fields below are filled in by later phases. They are null until then.
    category: str | None
    priority: str | None
    urgency: str | None
    impact: str | None
    assigned_team: str | None
    classification_confidence: float | None

    retrieved_documents: list | None
    suggested_resolution: str | None
    resolution_confidence: float | None

    auto_resolution_eligible: bool | None
    guardrail_status: str | None

    created_at: datetime
    updated_at: datetime


class StatusUpdate(BaseModel):
    """Request to move a ticket to a new status. The backend decides if the move is allowed."""

    model_config = ConfigDict(extra="forbid")

    status: TicketStatus
