from datetime import datetime
from enum import StrEnum
import uuid

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base
from app.models.ticket import utcnow


class AuditEventType(StrEnum):
    """Only events for features that exist. Later phases add their own (classification, routing, ...)."""

    TICKET_CREATED = "ticket_created"
    STATUS_CHANGED = "status_changed"
    PROCESSING_STARTED = "processing_started"


class AuditEvent(Base):
    """One entry in a ticket's history. Append-only: the app never updates or deletes these.

    The integer id increases with every insert, so it gives a strict order even if
    two events get the same timestamp.
    """

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("tickets.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(50))
    message: Mapped[str] = mapped_column(Text)
    # Small structured facts, e.g. {"from_status": "new", "to_status": "processing"}.
    # Called "details" because "metadata" is reserved by SQLAlchemy's declarative base.
    details: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
