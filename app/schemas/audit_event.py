from datetime import datetime
import uuid

from pydantic import BaseModel, ConfigDict, Field

from app.models import AuditEventType


class AuditEventRead(BaseModel):
    """One entry in a ticket's history."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="Increases with every event, so it also gives the order.")
    ticket_id: uuid.UUID
    event_type: AuditEventType
    message: str
    details: dict | None = Field(description="Structured facts about the event, if any.")
    created_at: datetime
