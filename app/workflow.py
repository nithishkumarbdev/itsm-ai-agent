"""What happens when a ticket is processed.

The steps are: start processing -> classify -> route -> retrieve knowledge -> generate a
grounded resolution suggestion. Later phases (guardrails) add their steps here, after
resolution.

The LLM only *proposes*: a classification, and later a resolution suggestion. This module
and the services it calls decide what is saved: only a validated classification is written,
the team comes from deterministic routing, retrieval only supplies evidence, a resolution is
only kept if every one of its cited sources was actually retrieved, and every outcome is
audited. The ticket is never marked resolved and no action is ever executed: a suggestion is
text for a human to review, not proof the issue is fixed.
"""

import logging
import time
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.audit import record_event
from app.db.database import SessionLocal
from app.embeddings.base import EmbeddingError
from app.lifecycle import change_status
from app.models import AuditEventType, Ticket, TicketStatus
from app.schemas.classification import TicketClassification
from app.schemas.resolution import TicketResolution
from app.services.classification import ClassificationError, ClassificationService
from app.services.resolution import ResolutionError, ResolutionService
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
    db: Session,
    ticket: Ticket,
    classifier: ClassificationService,
    retriever: RetrievalService,
    resolver: ResolutionService,
) -> Ticket:
    """Process one ticket. `ticket` must already be loaded (and locked) by the caller.

    Two short transactions with the slow external calls between them, so no database lock
    or transaction is held while waiting for the LLM or the embedding model:

      1. new/awaiting_human/escalated -> processing (+ audit), commit
      2. classify with the LLM, route by rules, embed the search query, search the knowledge
         index (its own short-lived session, unrelated to the ticket), and if evidence was
         found, generate a resolution suggestion with the LLM (no transaction open on the
         ticket at any point in this step)
      3. re-lock the ticket, save classification + team + evidence + resolution, write all
         audit events, commit

    Failure handling: if classification or routing fails, nothing new is saved. If retrieval
    fails, the valid classification and routing are kept, and resolution is not attempted. If
    only resolution fails, classification, routing and the retrieved evidence are kept.
    Whenever something fails, a failure event is written and the ticket moves to
    awaiting_human (the lifecycle's "a person must look at this" state, from which /process
    can be called again to retry). When retrieval finds no relevant evidence at all,
    resolution is skipped rather than attempted: no evidence means no invented answer, and
    that is not treated as a failure.
    """
    # --- 1. start processing -------------------------------------------------------
    change_status(db, ticket, TicketStatus.PROCESSING)
    record_event(db, ticket.id, AuditEventType.PROCESSING_STARTED, "Ticket processing started")
    db.commit()

    # --- 2. external work: classify, route, retrieve evidence, generate a resolution ---
    ticket_id = ticket.id
    title, description, source = ticket.title, ticket.description, ticket.source
    logger.info("classification started ticket=%s model=%s", ticket_id, classifier.llm_name)
    started = time.perf_counter()

    classification: TicketClassification | None = None
    decision: RoutingDecision | None = None
    outcome: RetrievalOutcome | None = None
    resolution: TicketResolution | None = None
    failure: Failure | None = None  # classification/routing failed: nothing new is saved
    retrieval_failure: Failure | None = None  # retrieval failed: classification/routing are kept
    resolution_failure: Failure | None = None  # resolution failed: classification/routing/evidence are kept

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
            # A separate, short-lived session: reading the knowledge index has nothing to do
            # with the ticket's own transaction, and this keeps the main session free while
            # the (potentially slow) resolution call below runs.
            with SessionLocal() as scratch:
                outcome = retriever.search(scratch, query_vector, category=classification.category.value)
        except EmbeddingError as exc:
            retrieval_failure = Failure(
                AuditEventType.RETRIEVAL_FAILED,
                f"Retrieval failed: {exc.kind}",
                exc.kind,
                {"provider_detail": exc.detail} if exc.detail else {},
                exc,
            )
        except KnowledgeIndexError as exc:
            retrieval_failure = Failure(
                AuditEventType.RETRIEVAL_FAILED, "Retrieval failed: index_incompatible", "index_incompatible", {}, exc
            )

    if failure is None and retrieval_failure is None and outcome.results:
        try:
            resolution = resolver.generate(
                title=title,
                description=description,
                source=source,
                category=classification.category.value,
                priority=classification.priority.value,
                urgency=classification.urgency.value,
                impact=classification.impact.value,
                evidence=outcome.results,
            )
        except ResolutionError as exc:
            resolution_failure = Failure(
                AuditEventType.RESOLUTION_FAILED,
                f"Resolution generation failed: {exc.kind}",
                exc.kind,
                exc.detail,
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
        if retrieval_failure is not None:
            _clear_retrieval_and_resolution(ticket)  # never leave evidence/resolution from an earlier run
            _save_failure(db, ticket, retrieval_failure)
            _log_failure(ticket_id, retrieval_failure, latency_ms)
        else:
            _save_retrieval(db, ticket, classification.category.value, outcome)
            if resolution_failure is not None:
                _clear_resolution(ticket)  # never leave a resolution from an earlier run next to fresh evidence
                _save_failure(db, ticket, resolution_failure)
                _log_failure(ticket_id, resolution_failure, latency_ms)
            elif resolution is not None:
                _save_resolution(db, ticket, resolution, outcome.results)
            else:
                _clear_resolution(ticket)  # no evidence: nothing was generated, and nothing stale is kept either
            logger.info(
                "processing completed ticket=%s category=%s team=%s confidence=%.2f retrieved=%d "
                "resolution=%s latency_ms=%d",
                ticket_id,
                classification.category.value,
                decision.assigned_team.value,
                classification.confidence,
                len(outcome.results),
                "generated" if resolution is not None else ("failed" if resolution_failure else "skipped_no_evidence"),
                latency_ms,
            )
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


def _save_resolution(db: Session, ticket: Ticket, resolution: TicketResolution, evidence: list) -> None:
    """Store the suggested resolution and only the sources it actually cited, and audit it.

    This never changes `status`: a suggestion is not proof the ticket is fixed, so the
    ticket stays in `processing` for a person to review.
    """
    ticket.suggested_resolution = resolution.suggested_resolution
    ticket.resolution_confidence = resolution.resolution_confidence
    cited_ids = {str(ref.chunk_id) for ref in resolution.source_references}
    cited = [chunk for chunk in evidence if str(chunk.chunk_id) in cited_ids]
    ticket.resolution_sources = [
        {
            "document_id": str(chunk.document_id),
            "document_title": chunk.document_title,
            "chunk_id": str(chunk.chunk_id),
            "chunk_index": chunk.chunk_index,
        }
        for chunk in cited
    ]
    record_event(
        db,
        ticket.id,
        AuditEventType.RESOLUTION_GENERATED,
        f"Resolution suggested, citing {len(cited)} source(s)"
        if cited
        else "Resolution suggested: evidence was insufficient",
        {
            "result_count": len(evidence),
            "resolution_confidence": resolution.resolution_confidence,
            "source_count": len(cited),
            "reason": resolution.reason,
        },
    )


def _clear_resolution(ticket: Ticket) -> None:
    ticket.suggested_resolution = None
    ticket.resolution_confidence = None
    ticket.resolution_sources = None


def _clear_retrieval_and_resolution(ticket: Ticket) -> None:
    ticket.retrieved_documents = None  # never leave evidence from an earlier run next to a new classification
    _clear_resolution(ticket)


def _save_failure(db: Session, ticket: Ticket, failure: Failure) -> None:
    record_event(
        db,
        ticket.id,
        failure.event_type,
        failure.message,
        {"error_type": failure.kind, **failure.detail},
    )
    change_status(db, ticket, TicketStatus.AWAITING_HUMAN)
