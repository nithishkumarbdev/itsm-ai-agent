from sqlalchemy.orm import Session

from app.audit import record_event
from app.models import AuditEventType, Ticket, TicketStatus

S = TicketStatus

# The whole lifecycle in one place. A move that is not listed here is illegal.
ALLOWED_TRANSITIONS: dict[TicketStatus, frozenset[TicketStatus]] = {
    S.NEW: frozenset({S.PROCESSING}),
    S.PROCESSING: frozenset({S.AWAITING_HUMAN, S.RESOLVED, S.ESCALATED}),
    S.AWAITING_HUMAN: frozenset({S.PROCESSING, S.RESOLVED, S.ESCALATED}),
    S.ESCALATED: frozenset({S.PROCESSING, S.CLOSED}),
    S.RESOLVED: frozenset({S.CLOSED}),
    S.CLOSED: frozenset(),  # terminal
}


class InvalidTransition(Exception):
    def __init__(self, current: TicketStatus, target: TicketStatus):
        super().__init__(f"Invalid ticket status transition: {current} -> {target}")
        self.current = current
        self.target = target


def can_transition(current: TicketStatus, target: TicketStatus) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def change_status(db: Session, ticket: Ticket, target: TicketStatus) -> None:
    """Move a ticket to a new status and record it. Raises InvalidTransition if not allowed.

    Does not commit. The caller commits, so the status change and its audit event
    are written in the same transaction.
    """
    current = TicketStatus(ticket.status)
    if not can_transition(current, target):
        raise InvalidTransition(current, target)

    ticket.status = target.value
    record_event(
        db,
        ticket.id,
        AuditEventType.STATUS_CHANGED,
        f"Ticket moved from {current} to {target}",
        {"from_status": current.value, "to_status": target.value},
    )
