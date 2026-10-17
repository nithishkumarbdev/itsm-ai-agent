"""What happens when a ticket is processed.

Today the steps are: start processing -> classify -> route. Later phases (retrieval,
resolution, guardrails) add their steps here, after routing.

The LLM only *proposes* a classification. This module and the rules it calls decide what
is saved: only a validated classification is written, the team comes from deterministic
routing, and every outcome is audited.
"""

import logging
import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.audit import record_event
from app.lifecycle import change_status
from app.models import AuditEventType, Ticket, TicketStatus
from app.schemas.classification import TicketClassification
from app.services.classification import ClassificationError, ClassificationService
from app.services.routing import RoutingDecision, route_ticket

logger = logging.getLogger(__name__)


@dataclass
class Failure:
    """A step that did not produce a usable result. Nothing from it is saved."""

    event_type: AuditEventType
    message: str
    kind: str
    detail: dict
    cause: BaseException


def run_processing(db: Session, ticket: Ticket, classifier: ClassificationService) -> Ticket:
    """Process one ticket. `ticket` must already be loaded (and locked) by the caller.

    Two short transactions with the LLM call between them, so no database lock or
    transaction is held while waiting for the model:

      1. new/awaiting_human/escalated -> processing (+ audit), commit
      2. classify with the LLM, then route by deterministic rules (no transaction open)
      3. re-lock the ticket, save classification + team + both audit events, commit

    Classification and routing are saved together or not at all. If either step fails,
    nothing is invented or half-saved: the failure is audited and the ticket moves to
    awaiting_human (the lifecycle's "a person must look at this" state, from which
    /process can be called again to retry).
    """
    # --- 1. start processing -------------------------------------------------------
    change_status(db, ticket, TicketStatus.PROCESSING)
    record_event(db, ticket.id, AuditEventType.PROCESSING_STARTED, "Ticket processing started")
    db.commit()

    # --- 2. classify (LLM proposes), then route (backend decides) ------------------
    ticket_id = ticket.id
    logger.info("classification started ticket=%s model=%s", ticket_id, classifier.llm_name)
    started = time.perf_counter()
    classification: TicketClassification | None = None
    decision: RoutingDecision | None = None
    failure: Failure | None = None
    try:
        classification = classifier.classify(
            title=ticket.title, description=ticket.description, source=ticket.source
        )
    except ClassificationError as exc:
        failure = Failure(
            AuditEventType.CLASSIFICATION_FAILED,
            f"Classification failed: {exc.kind}",
            exc.kind,
            exc.detail,
            exc,
        )
    else:
        try:
            decision = route_ticket(classification.category)
        except Exception as exc:  # routing is a pure function and should never raise
            classification = None  # do not keep a classification that has no routing
            failure = Failure(
                AuditEventType.ROUTING_FAILED,
                "Routing failed: routing_error",
                "routing_error",
                {"exception": type(exc).__name__},
                exc,
            )
    latency_ms = round((time.perf_counter() - started) * 1000)

    # --- 3. save the outcome -----------------------------------------------------------
    db.refresh(ticket, with_for_update=True)  # fresh state, locked again
    if ticket.status != TicketStatus.PROCESSING:
        # Someone changed the ticket while the LLM was running. Do not overwrite their change.
        logger.warning("result discarded ticket=%s: status is now %s", ticket_id, ticket.status)
        db.rollback()  # release the lock, save nothing
        db.refresh(ticket)  # rollback expired the object; reload so it can be returned
        return ticket

    if failure is None:
        _save_result(db, ticket, classification, decision)
        logger.info(
            "processing completed ticket=%s category=%s team=%s confidence=%.2f latency_ms=%d",
            ticket_id,
            classification.category.value,
            decision.assigned_team.value,
            classification.confidence,
            latency_ms,
        )
    else:
        _save_failure(db, ticket, failure)
        logger.error(
            "processing failed ticket=%s step=%s kind=%s detail=%s latency_ms=%d",
            ticket_id,
            failure.event_type.value,
            failure.kind,
            failure.detail,
            latency_ms,
            # Provider errors carry a useful traceback. Validation errors can echo model
            # output, so they are logged without one.
            exc_info=failure.cause if failure.kind != "validation_error" else False,
        )
    db.commit()
    db.refresh(ticket)
    return ticket


def _save_result(
    db: Session, ticket: Ticket, result: TicketClassification, decision: RoutingDecision
) -> None:
    ticket.category = result.category.value
    ticket.priority = result.priority.value
    ticket.urgency = result.urgency.value
    ticket.impact = result.impact.value
    ticket.classification_confidence = result.confidence
    ticket.classification_reason = result.reason
    record_event(
        db,
        ticket.id,
        AuditEventType.CLASSIFICATION_COMPLETED,
        f"Ticket classified as {result.category.value} ({result.priority.value})",
        result.model_dump(mode="json"),
    )

    ticket.assigned_team = decision.assigned_team.value
    record_event(
        db,
        ticket.id,
        AuditEventType.ROUTING_DECIDED,
        decision.reason,
        {
            "category": result.category.value,
            "assigned_team": decision.assigned_team.value,
            "rule": decision.rule,
            "reason": decision.reason,
        },
    )


def _save_failure(db: Session, ticket: Ticket, failure: Failure) -> None:
    record_event(
        db,
        ticket.id,
        failure.event_type,
        failure.message,
        {"error_type": failure.kind, **failure.detail},
    )
    change_status(db, ticket, TicketStatus.AWAITING_HUMAN)
