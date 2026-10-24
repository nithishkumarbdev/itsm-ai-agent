"""The tools that exist. This module IS the allow-list: a name that is not in TOOLS cannot be run.

Deliberately absent: anything generic or unrestricted (run a shell command, run SQL, execute
code, delete data) and a generic "update these ticket fields" tool. Each tool below does one
narrow thing, validates its own arguments, and (for the two that change state) is subject to a
backend policy that does not depend on who is asking or what they say.

Tool RESULTS are untrusted data, like ticket text: they contain user-written or knowledge-base
text. If a result is ever given to an LLM, it must go in a labelled data block, never into
instructions (see how app/services/resolution.py builds its prompt).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.audit import record_event
from app.guardrails.policy import evaluate as evaluate_guardrails
from app.models import AuditEventType, Ticket, TicketStatus
from app.schemas.classification import Category
from app.services.routing import Team
from app.tools.base import OPERATOR, ToolContext, ToolDenied, ToolSpec

# Tickets a person may still be re-assigned: not new (never classified), resolved or closed.
ASSIGNABLE_STATUSES = frozenset(
    {TicketStatus.PROCESSING.value, TicketStatus.AWAITING_HUMAN.value, TicketStatus.ESCALATED.value}
)


def _meaningful(value: str) -> str:
    if not any(ch.isalnum() for ch in value):
        raise ValueError("must contain at least one letter or digit")
    return value


# --------------------------------------------------------------------------- get_ticket


class NoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _get_ticket(ctx: ToolContext, args: NoArguments) -> dict:
    """A deliberately small view of the ticket. Not included: the requester's email (personal
    data no tool needs), the retrieved-evidence list, and internal timestamps."""
    t = ctx.ticket
    return {
        "id": str(t.id),
        "title": t.title,
        "description": t.description,
        "status": t.status,
        "category": t.category,
        "priority": t.priority,
        "urgency": t.urgency,
        "impact": t.impact,
        "assigned_team": t.assigned_team,
        "guardrail_status": t.guardrail_status,
        "suggested_resolution": t.suggested_resolution,
        "resolution_confidence": t.resolution_confidence,
    }


# --------------------------------------------------------------------------- search_knowledge_base


class SearchKnowledgeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=500)
    top_k: int | None = Field(default=None, ge=1, le=10)
    category: Category | None = None

    @field_validator("query")
    @classmethod
    def query_must_be_meaningful(cls, value: str) -> str:
        return _meaningful(value)


def _search_knowledge_base(ctx: ToolContext, args: SearchKnowledgeArguments) -> dict:
    outcome = ctx.retriever.retrieve(
        ctx.db, args.query, top_k=args.top_k, category=args.category.value if args.category else None
    )
    return {
        "results": [
            {
                "document_title": r.document_title,
                "section": r.section,
                "content": r.content,
                "similarity_score": r.similarity_score,
                "chunk_id": str(r.chunk_id),
            }
            for r in outcome.results
        ],
        "candidates_considered": outcome.candidates_considered,
    }


# --------------------------------------------------------------------------- check_resolution_policy


def _check_policy_policy(ticket: Ticket, args: NoArguments) -> None:
    if ticket.category is None:
        raise ToolDenied("ticket_not_classified")


def _check_resolution_policy(ctx: ToolContext, args: NoArguments) -> dict:
    """Re-run the guardrail check on what is stored on the ticket right now, and compare it
    with the verdict recorded when the ticket was processed."""
    t = ctx.ticket
    has_resolution = t.suggested_resolution is not None
    result = evaluate_guardrails(
        category=t.category,
        ticket_text=f"{t.title}\n{t.description}",
        resolution_text=t.suggested_resolution,
        source_count=len(t.resolution_sources or []) if has_resolution else None,
    )
    return {
        "status": result.status.value,
        "stored_status": t.guardrail_status,
        "security_category": result.security_category,
        "grounded": result.grounded,
        "prompt_injection_suspected": result.prompt_injection_suspected,
        "blocked_reasons": result.blocked_reasons,
    }


# --------------------------------------------------------------------------- add_ticket_note


class AddNoteArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    note: str = Field(min_length=1, max_length=1000)

    @field_validator("note")
    @classmethod
    def note_must_be_meaningful(cls, value: str) -> str:
        return _meaningful(value)


def _add_note_policy(ticket: Ticket, args: AddNoteArguments) -> None:
    if ticket.status == TicketStatus.CLOSED.value:
        raise ToolDenied("ticket_closed")


def _add_ticket_note(ctx: ToolContext, args: AddNoteArguments) -> dict:
    """Notes are stored as append-only audit events, so they are ordered with everything else
    that happened to the ticket and can never be edited or deleted."""
    record_event(ctx.db, ctx.ticket.id, AuditEventType.TICKET_NOTE_ADDED, args.note, {"actor": ctx.actor})
    return {"recorded": True}


# --------------------------------------------------------------------------- assign_ticket


class AssignTicketArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    team: Team


def _assign_policy(ticket: Ticket, args: AssignTicketArguments) -> None:
    if ticket.category is None:
        raise ToolDenied("ticket_not_classified")  # process it first: routing decides the first team
    if ticket.status not in ASSIGNABLE_STATUSES:
        raise ToolDenied("status_not_assignable")
    if ticket.category == Category.SECURITY.value and args.team != Team.SECURITY:
        # Same precedence rule as routing: a security ticket always stays with Security.
        raise ToolDenied("security_tickets_stay_with_security")


def _assign_ticket(ctx: ToolContext, args: AssignTicketArguments) -> dict:
    previous = ctx.ticket.assigned_team
    changed = previous != args.team.value
    if changed:
        ctx.ticket.assigned_team = args.team.value
    return {"assigned_team": args.team.value, "previous_team": previous, "changed": changed}


# --------------------------------------------------------------------------- the registry

TOOLS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            name="get_ticket",
            description="Read a small, non-personal view of the ticket being worked on.",
            args_model=NoArguments,
            handler=_get_ticket,
            audit_details=lambda args, data: {},
        ),
        ToolSpec(
            name="search_knowledge_base",
            description="Search the internal knowledge base. Read-only, active documents only.",
            args_model=SearchKnowledgeArguments,
            handler=_search_knowledge_base,
            audit_details=lambda args, data: {
                # The query text is not recorded (it can contain ticket content), only its shape.
                "query_length": len(args.query),
                "top_k": args.top_k,
                "category": args.category.value if args.category else None,
                "result_count": len(data["results"]),
            },
            needs_retriever=True,
        ),
        ToolSpec(
            name="check_resolution_policy",
            description="Re-run the guardrail check on the ticket's stored resolution. Read-only.",
            args_model=NoArguments,
            handler=_check_resolution_policy,
            audit_details=lambda args, data: {
                "status": data["status"],
                "matches_stored_status": data["status"] == data["stored_status"],
            },
            policy=_check_policy_policy,
        ),
        ToolSpec(
            name="add_ticket_note",
            description="Add a note to the ticket's history. Append-only.",
            args_model=AddNoteArguments,
            handler=_add_ticket_note,
            audit_details=lambda args, data: {"note_length": len(args.note)},
            policy=_add_note_policy,
            writes=True,
        ),
        ToolSpec(
            name="assign_ticket",
            description="Reassign the ticket to a different team. Operators only.",
            args_model=AssignTicketArguments,
            handler=_assign_ticket,
            audit_details=lambda args, data: {
                "team": data["assigned_team"],
                "previous_team": data["previous_team"],
                "changed": data["changed"],
            },
            policy=_assign_policy,
            writes=True,
            allowed_actors=frozenset({OPERATOR}),  # an automated caller can never reassign
        ),
    )
}
