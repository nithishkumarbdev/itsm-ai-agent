import uuid
from datetime import datetime, timezone
from enum import StrEnum

from sqlalchemy import JSON, Boolean, DateTime, Float, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TicketStatus(StrEnum):
    """Lifecycle states. Which moves between them are legal is defined in app/lifecycle.py."""

    NEW = "new"
    PROCESSING = "processing"
    AWAITING_HUMAN = "awaiting_human"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"


class Ticket(Base):
    """An IT support ticket.

    Only the intake fields (source .. description) and status are used so far.
    Every other column is nullable and stays NULL until the phase that owns it:
    classification (3), routing (4), retrieval/resolution (5-6), guardrails (7),
    auto-resolution policy (9).
    """

    __tablename__ = "tickets"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)

    # --- intake (set at creation) ---
    source: Mapped[str] = mapped_column(String(20))
    requester: Mapped[str] = mapped_column(String(320))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(30), default=TicketStatus.NEW.value, index=True)

    # --- classification / routing (NULL = not done yet) ---
    category: Mapped[str | None] = mapped_column(String(50))
    priority: Mapped[str | None] = mapped_column(String(10))
    urgency: Mapped[str | None] = mapped_column(String(20))
    impact: Mapped[str | None] = mapped_column(String(20))
    assigned_team: Mapped[str | None] = mapped_column(String(100))
    classification_confidence: Mapped[float | None] = mapped_column(Float)

    # --- retrieval / resolution ---
    retrieved_documents: Mapped[list | None] = mapped_column(JSON)
    suggested_resolution: Mapped[str | None] = mapped_column(Text)
    resolution_confidence: Mapped[float | None] = mapped_column(Float)

    # --- policy / safety (NULL = not evaluated, which is different from False) ---
    auto_resolution_eligible: Mapped[bool | None] = mapped_column(Boolean)
    guardrail_status: Mapped[str | None] = mapped_column(String(30))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
