"""What happens when a ticket is processed.

Today the steps are: start processing -> classify -> route -> retrieve knowledge.
Later phases (resolution, guardrails) add their steps here, after retrieval.

The LLM only *proposes* a classification. This module and the rules it calls decide what
is saved: only a validated classification is written, the team comes from deterministic
routing, retrieval only supplies evidence, and every outcome is audited.
"""

import logging
import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.audit import record_event
from app.embeddings.base import EmbeddingError
from app.lifecycle import change_status
from app.models import AuditEventType, Ticket, TicketStatus
from app.schemas.classification import TicketClassification
from app.services.classification import ClassificationError, ClassificationService
from app.services.retrieval import KnowledgeIndexError, RetrievalOutcome, RetrievalService, build_query
from app.services.routing import RoutingDecision, route_ticket

logger = logging.getLogger(__name__)


@dataclass
class Failure:
    """A step that did not produce a usable result."""

    event_type: AuditEventType
    message: str
    kind: str
    detail: dict
    cause: BaseException


def run_processing(
    db: Session, ticket: Ticket, classifier: ClassificationService, retriever: RetrievalService
) -> Ticket:
    """Process one ticket. `ticket` must already be loaded (and locked) by the caller.

    Two short transactions with the slow external calls between them, so no database lock
    or transaction is held while waiting for the LLM or the embedding model:

      1. new/awaiting_human/escalated -> processing (+ audit), commit
      2. classify with the LLM, route by rules, embed the search query (no transaction open)
      3. re-lock the ticket, save classification + team, search the knowledge index,
         save the evidence, write all audit events, commit

    Failure handling: if classification or routing fails, nothing new is saved. If only
    retrieval fails, the valid classification and routing are kept. Either way a failure
    event is written and the ticket moves to awaiting_human (the lifecycle's "a person must
    look at this" state, from which /process can be called again to retry).
    """
    # --- 1. start processing -------------------------------------------------------
    change_status(db, ticket, TicketStatus.PROCESSING)
    record_event(db, ticket.id, AuditEventType.PROCESSING_STARTED, "Ticket processing started")
    db.commit()

    # --- 2. external work: classify, route, embed the query ------------------------
    ticket_id = ticket.id
    title, description, source = ticket.title, ticket.description, ticket.source
    logger.info("classification started ticket=%s model=%s", ticket_id, classifier.llm_name)
    started = time.perf_counter()
    classification: TicketClassification | None = None
    decision: RoutingDecision | None = None
    query_vector: list[float] | None = None
    failure: Failure | None = None  # classification/routing failed: nothing new is saved
    retrieval_failure: Failure | None = None  # retrieval failed: classification and routing are kept
    try:
        classification = classifier.classify(title=title, description=description, source=source)
    except ClassificationError as exc:
        failure = Failure(
            AuditEventType.CLASSIFICATION_FAILED, f"Classification failed: {exc.kind}", exc.kind, exc.detail, exc
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
    if failure is None:
        try:
            query_vector = retriever.embed_query(build_query(title, description))
        except EmbeddingError as exc:
            retrieval_failure = Failure(
                AuditEventType.RETRIEVAL_FAILED,
                f"Retrieval failed: {exc.kind}",
                exc.kind,
                {"provider_detail": exc.detail} if exc.detail else {},
                exc,
            )
    latency_ms = round((time.perf_counter() - started) * 1000)

    # --- 3. save the outcome -----------------------------------------------------------
    db.refresh(ticket, with_for_update=True)  # fresh state, locked again
    if ticket.status != TicketStatus.PROCESSING:
        # Someone changed the ticket while the slow calls were running. Do not overwrite their change.
        logger.warning("result discarded ticket=%s: status is now %s", ticket_id, ticket.status)
        db.rollback()  # release the lock, save nothing
        db.refresh(ticket)  # rollback expired the object; reload so it can be returned
        return ticket

    if failure is not None:
        _save_failure(db, ticket, failure)
        _log_failure(ticket_id, failure, latency_ms)
    else:
        _save_result(db, ticket, classification, decision)
        outcome: RetrievalOutcome | None = None
        if retrieval_failure is None:
            try:
                outcome = retriever.search(db, query_vector, category=classification.category.value)
            except KnowledgeIndexError as exc:
                retrieval_failure = Failure(
                    AuditEventType.RETRIEVAL_FAILED, "Retrieval failed: index_incompatible", "index_incompatible", {}, exc
                )
        if outcome is not None:
            _save_retrieval(db, ticket, classification.category.value, outcome)
            logger.info(
                "processing completed ticket=%s category=%s team=%s confidence=%.2f retrieved=%d latency_ms=%d",
                ticket_id,
                classification.category.value,
                decision.assigned_team.value,
                classification.confidence,
                len(outcome.results),
                latency_ms,
            )
        else:
            ticket.retrieved_documents = None  # never leave evidence from an earlier run next to a new classification
            _save_failure(db, ticket, retrieval_failure)
            _log_failure(ticket_id, retrieval_failure, latency_ms)
    db.commit()
    db.refresh(ticket)
    return ticket


def _log_failure(ticket_id, failure: Failure, latency_ms: int) -> None:
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


def _save_retrieval(db: Session, ticket: Ticket, category: str, outcome: RetrievalOutcome) -> None:
    """Store references to the retrieved chunks (ids and scores, not their text) and audit it."""
    ticket.retrieved_documents = [r.reference() for r in outcome.results]  # [] = searched, nothing relevant
    titles = list(dict.fromkeys(r.document_title for r in outcome.results))  # unique, best first
    count = len(outcome.results)
    record_event(
        db,
        ticket.id,
        AuditEventType.KNOWLEDGE_RETRIEVED,
        f"Retrieved {count} knowledge chunk(s)" if count else "No sufficiently relevant knowledge found",
        {
            "result_count": count,
            "top_score": max((r.similarity_score for r in outcome.results), default=None),
            "category": category,
            "min_score": outcome.min_score,
            "candidates_considered": outcome.candidates_considered,
            "documents": titles,
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
