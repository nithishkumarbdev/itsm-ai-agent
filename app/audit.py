import uuid

from sqlalchemy.orm import Session

from app.models import AuditEvent, AuditEventType


def record_event(
    db: Session,
    ticket_id: uuid.UUID,
    event_type: AuditEventType,
    message: str,
    details: dict | None = None,
) -> None:
    """Add an audit event to the current transaction.

    It deliberately does not commit: the caller commits the event together with
    the change it describes, so the two succeed or fail as one.
    """
    db.add(
        AuditEvent(
            ticket_id=ticket_id,
            event_type=event_type.value,
            message=message,
            details=details,
        )
    )
