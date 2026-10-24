"""Shared types for the controlled tool layer.

A tool is a narrowly scoped operation on ONE ticket that something other than a plain API
request wants to perform (a later workflow step, or an LLM asked to pick tools). Every tool is
described by a ToolSpec, and all of them run through execute_tool (app/tools/executor.py), which
is the only way to run one. That is what makes each call validated, policy-checked, logged and
audited: there is no other path to a tool's handler.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.models import Ticket
from app.services.retrieval import RetrievalService

AGENT = "agent"  # an automated caller (for example an LLM choosing tools): the most restricted actor
OPERATOR = "operator"  # a human on the IT team, acting through trusted backend code
ACTORS = frozenset({AGENT, OPERATOR})
# There is no authentication yet, so `actor` is a label asserted by the (trusted, internal) caller,
# recorded in the audit trail. It is not proof of identity.


class ToolDenied(Exception):
    """Raised by a tool's policy function to refuse a call. `reason` is a short machine-readable
    code (never free text from the caller), stored in the audit trail."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ToolContext:
    db: Session
    ticket: Ticket
    actor: str
    retriever: RetrievalService | None


def allow_always(ticket: Ticket, args: Any) -> None:
    """The default policy: no ticket-specific restriction."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]  # strict: unknown fields are rejected
    handler: Callable[[ToolContext, Any], dict]  # does the work; returns the data given to the caller
    audit_details: Callable[[Any, dict], dict]  # what to record about a successful call (never secrets)
    policy: Callable[[Ticket, Any], None] = allow_always  # raises ToolDenied to refuse
    writes: bool = False  # True: the ticket row is locked for the call
    allowed_actors: frozenset[str] = ACTORS
    needs_retriever: bool = False


@dataclass(frozen=True)
class ToolOutcome:
    status: Literal["ok", "denied", "failed", "ticket_not_found"]
    tool: str
    reason: str | None = None  # machine-readable code for denied / failed / not found
    data: dict | None = None  # only for "ok"
