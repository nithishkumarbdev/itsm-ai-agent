import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_classification_service, get_retrieval_service
from app.audit import record_event
from app.db.database import get_db
from app.lifecycle import change_status
from app.models import AuditEvent, AuditEventType, Ticket
from app.schemas.audit_event import AuditEventRead
from app.schemas.ticket import StatusUpdate, TicketCreate, TicketRead
from app.services.classification import ClassificationService
from app.services.retrieval import RetrievalService
from app.workflow import run_processing

router = APIRouter(prefix="/tickets", tags=["tickets"])

NOT_FOUND = {404: {"description": "Ticket not found"}}
INVALID_TRANSITION = {409: {"description": "Status transition not allowed"}}


def get_ticket_or_404(db: Session, ticket_id: uuid.UUID, *, lock: bool = False) -> Ticket:
    """Load a ticket. With lock=True the row is locked until the transaction ends
    (SELECT ... FOR UPDATE), so two concurrent requests cannot both act on the same
    old status."""
    ticket = db.get(Ticket, ticket_id, with_for_update=True if lock else None)
    if ticket is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return ticket


@router.post("", response_model=TicketRead, status_code=status.HTTP_201_CREATED)
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    """Ticket intake. The ticket and its `ticket_created` event are saved in one transaction."""
    ticket = Ticket(**payload.model_dump())
    db.add(ticket)
    db.flush()  # sends the INSERT so the ticket id exists; nothing is committed yet
    record_event(
        db,
        ticket.id,
        AuditEventType.TICKET_CREATED,
        "Ticket created",
        {"source": ticket.source},
    )
    db.commit()
    db.refresh(ticket)
    return ticket


@router.get("", response_model=list[TicketRead])
def list_tickets(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List tickets, newest first."""
    stmt = select(Ticket).order_by(Ticket.created_at.desc(), Ticket.id.desc())
    return db.scalars(stmt.limit(limit).offset(offset)).all()


@router.get("/{ticket_id}", response_model=TicketRead, responses=NOT_FOUND)
def get_ticket(ticket_id: uuid.UUID, db: Session = Depends(get_db)):
    return get_ticket_or_404(db, ticket_id)


@router.post(
    "/{ticket_id}/status",
    response_model=TicketRead,
    responses={**NOT_FOUND, **INVALID_TRANSITION},
)
def update_ticket_status(ticket_id: uuid.UUID, payload: StatusUpdate, db: Session = Depends(get_db)):
    """Move a ticket to another status, only if the lifecycle allows it (see app/lifecycle.py)."""
    ticket = get_ticket_or_404(db, ticket_id, lock=True)
    change_status(db, ticket, payload.status)
    db.commit()
    db.refresh(ticket)
    return ticket


@router.post(
    "/{ticket_id}/process",
    response_model=TicketRead,
    responses={**NOT_FOUND, **INVALID_TRANSITION},
)
def process_ticket(
    ticket_id: uuid.UUID,
    db: Session = Depends(get_db),
    classifier: ClassificationService = Depends(get_classification_service),
    retriever: RetrievalService = Depends(get_retrieval_service),
):
    """Process a ticket: `processing`, then LLM classification, rule-based routing, and knowledge retrieval.

    The LLM only proposes the classification. The team (`assigned_team`) is chosen by
    deterministic backend rules from the validated category; clients cannot set it.
    Retrieval then finds the internal knowledge most relevant to the ticket and stores
    references to it in `retrieved_documents` (`[]` means nothing was relevant enough).
    Retrieval provides evidence only: it does not change the routing or write an answer.

    Success: the ticket is returned with its classification, team and retrieved references,
    and stays in `processing`.

    If classification or routing fails, the response is still 200 but the ticket has no new
    classification and its status is `awaiting_human`. If only retrieval fails, the valid
    classification and routing are kept, `retrieved_documents` is null, and the status is
    `awaiting_human`. The reason is in the audit events (`classification_failed`,
    `routing_failed` or `retrieval_failed`). Nothing is resolved automatically.
    """
    ticket = get_ticket_or_404(db, ticket_id, lock=True)
    return run_processing(db, ticket, classifier, retriever)


@router.get("/{ticket_id}/events", response_model=list[AuditEventRead], responses=NOT_FOUND)
def list_ticket_events(ticket_id: uuid.UUID, db: Session = Depends(get_db)):
    """The ticket's history, oldest first."""
    get_ticket_or_404(db, ticket_id)
    stmt = (
        select(AuditEvent)
        .where(AuditEvent.ticket_id == ticket_id)
        .order_by(AuditEvent.created_at, AuditEvent.id)
    )
    return db.scalars(stmt).all()
