"""The only way to run a tool.

    look up tool -> load ticket -> actor allowed? -> arguments valid? -> policy allows?
        -> run handler (in a savepoint) -> audit the outcome

Every path ends in exactly one audit event on the ticket (tool_executed, tool_denied or
tool_failed), so "what did this tool call do, and why was it refused?" is always answerable.
Like the rest of the app, this never commits: the caller commits, so the tool's effect and
its audit event succeed or fail together.
"""

import logging
import time
import uuid

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.audit import record_event
from app.embeddings.base import EmbeddingError
from app.models import AuditEventType, Ticket
from app.services.retrieval import KnowledgeIndexError, RetrievalService
from app.tools.base import ACTORS, ToolContext, ToolDenied, ToolOutcome
from app.tools.definitions import TOOLS

logger = logging.getLogger(__name__)

MAX_NAME_CHARS = 100  # a tool name from an untrusted caller is stored in the audit trail: bound it


def execute_tool(
    db: Session,
    *,
    ticket_id: uuid.UUID,
    tool_name: str,
    arguments: dict,
    actor: str,
    retriever: RetrievalService | None = None,
) -> ToolOutcome:
    """Run one tool call against one ticket. `arguments` and `tool_name` are untrusted."""
    started = time.perf_counter()
    name = str(tool_name)[:MAX_NAME_CHARS]
    spec = TOOLS.get(tool_name) if isinstance(tool_name, str) else None

    # Write tools lock the ticket row until the caller's transaction ends, like every other
    # state change in the app, so two callers cannot act on the same old state.
    ticket = db.get(Ticket, ticket_id, with_for_update=True if (spec and spec.writes) else None)
    if ticket is None:
        logger.warning("tool call for unknown ticket tool=%s", name)
        return ToolOutcome("ticket_not_found", name, "ticket_not_found")

    def deny(reason: str, **extra) -> ToolOutcome:
        record_event(
            db, ticket_id, AuditEventType.TOOL_DENIED, f"Tool {name} denied: {reason}",
            {"tool": name, "actor": str(actor)[:MAX_NAME_CHARS], "reason": reason, **extra},
        )
        logger.warning("tool denied tool=%s actor=%s reason=%s ticket=%s", name, actor, reason, ticket_id)
        return ToolOutcome("denied", name, reason)

    if spec is None:
        return deny("unknown_tool")
    if actor not in ACTORS:
        return deny("unknown_actor")
    if actor not in spec.allowed_actors:
        return deny("actor_not_permitted")

    try:
        args = spec.args_model.model_validate(arguments)
    except ValidationError as exc:
        # Which fields were wrong, never their values (they are untrusted and may be sensitive).
        fields = sorted({str(e["loc"][0]) if e["loc"] else "(root)" for e in exc.errors()})
        return deny("invalid_arguments", fields=fields)

    try:
        spec.policy(ticket, args)
    except ToolDenied as exc:
        return deny(exc.reason)

    if spec.needs_retriever and retriever is None:
        return _fail(db, ticket_id, name, actor, "retrieval_not_configured")

    try:
        # A savepoint: if the handler fails half way, only ITS changes are undone, and the
        # failure itself can still be recorded in the caller's transaction.
        with db.begin_nested():
            data = spec.handler(ToolContext(db, ticket, actor, retriever), args)
            details = spec.audit_details(args, data)
    except EmbeddingError as exc:
        return _fail(db, ticket_id, name, actor, f"embedding_{exc.kind}")
    except KnowledgeIndexError:
        return _fail(db, ticket_id, name, actor, "index_incompatible")
    except Exception as exc:  # a tool must never take the caller down with it
        logger.exception("tool crashed tool=%s ticket=%s", name, ticket_id)
        return _fail(db, ticket_id, name, actor, "tool_error", exception=type(exc).__name__)

    record_event(
        db, ticket_id, AuditEventType.TOOL_EXECUTED, f"Tool {name} executed by {actor}",
        {"tool": name, "actor": actor, **details},
    )
    logger.info(
        "tool executed tool=%s actor=%s ticket=%s latency_ms=%d",
        name, actor, ticket_id, round((time.perf_counter() - started) * 1000),
    )
    return ToolOutcome("ok", name, data=data)


def _fail(db: Session, ticket_id: uuid.UUID, name: str, actor: str, reason: str, **extra) -> ToolOutcome:
    """Record a tool failure. Only a short code (and an exception class name) is stored, never
    a message, since messages can echo untrusted input or internal details."""
    record_event(
        db, ticket_id, AuditEventType.TOOL_FAILED, f"Tool {name} failed: {reason}",
        {"tool": name, "actor": actor, "reason": reason, **extra},
    )
    logger.error("tool failed tool=%s actor=%s reason=%s ticket=%s", name, actor, reason, ticket_id)
    return ToolOutcome("failed", name, reason)
