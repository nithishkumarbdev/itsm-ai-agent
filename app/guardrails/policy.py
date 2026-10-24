"""Deterministic safety checks, run after resolution generation.

This is backend policy, the same way routing is: no LLM call, no database, no randomness.
It can actively BLOCK a resolution whose text recommends something unsafe, and it always
records a policy signal (`guardrail_status`) for later phases to use when deciding
auto-resolution eligibility. Phase 8 does not implement that eligibility decision itself:
it only classifies and records the safety state, exactly as classification records category
without deciding routing, and routing decides a team without deciding automation.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum


class GuardrailStatus(StrEnum):
    PASSED = "passed"  # a resolution was generated, grounded in real evidence, and its text is safe
    REQUIRES_HUMAN_REVIEW = "requires_human_review"  # security-category tickets, always
    UNGROUNDED = "ungrounded"  # no resolution, or a resolution grounded in no cited evidence
    BLOCKED_RESTRICTED_CONTENT = "blocked_restricted_content"  # the resolution text itself is unsafe


# Phrases a resolution must never contain. Matched case-insensitively as plain patterns, not
# by a content classifier: a deterministic, auditable blocklist, the same kind of backend
# check as routing's category table. The resolution prompt already forbids all of this
# (see app/services/resolution.py); this is the backend's independent re-check, the same
# defense-in-depth stance as the source-reference check in ResolutionService.generate.
_RESTRICTED_ACTION_PATTERNS = [
    r"disable (the )?mfa", r"turn off (the )?mfa", r"disable multi-factor",
    r"disable (the )?firewall", r"turn off (the )?firewall",
    r"disable (the )?antivirus", r"turn off (the )?antivirus",
    r"disable (the )?security", r"bypass (the )?security",
    r"delete (the|your|this) account", r"remove (the|your|this) account",
    r"share your password", r"send (me|us) your password", r"give (me|us) your password",
    r"grant (you |yourself )?admin", r"give (you |yourself )?admin access", r"elevate .* to admin",
]

# Language that claims a completed action rather than a suggestion.
_COMPLETED_ACTION_PATTERNS = [
    r"issue (is|has been) (now )?(fixed|resolved)",
    r"(has|have) been (fixed|resolved|repaired|restored)",
    r"\bi (have )?(performed|executed|changed|accessed|deleted|reset|disabled|fixed|resolved|restored)\b",
]

# Common prompt-injection phrasing found in TICKET text. Detecting this is informational
# only: ticket text is already treated as untrusted data everywhere it reaches an LLM
# (see app/services/classification.py and app/services/resolution.py), so this does not by
# itself change how the ticket is processed or block anything.
_INJECTION_PATTERNS = [
    r"ignore (all )?(the )?(previous|prior|above) instructions",
    r"disregard (all )?(the )?(previous|prior|above) instructions",
    r"reveal your (system prompt|instructions)",
    r"you are now an? (admin|administrator)",
    r"new instructions?:",
]

_restricted_re = [re.compile(p, re.IGNORECASE) for p in _RESTRICTED_ACTION_PATTERNS]
_completed_re = [re.compile(p, re.IGNORECASE) for p in _COMPLETED_ACTION_PATTERNS]
_injection_re = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def unsafe_resolution_reasons(text: str) -> list[str]:
    """Which unsafe patterns, if any, appear in a resolution's text. Empty means safe."""
    reasons = [f"restricted_action:{p.pattern}" for p in _restricted_re if p.search(text)]
    reasons += [f"completed_action_claim:{p.pattern}" for p in _completed_re if p.search(text)]
    return reasons


def detect_prompt_injection(text: str) -> bool:
    """True if ticket text contains common instruction-override phrasing."""
    return any(pattern.search(text) for pattern in _injection_re)


@dataclass(frozen=True)
class GuardrailResult:
    status: GuardrailStatus
    security_category: bool
    grounded: bool | None  # None = no resolution was attempted (no evidence retrieved)
    prompt_injection_suspected: bool
    blocked_reasons: list[str] = field(default_factory=list)


def evaluate(
    *, category: str, ticket_text: str, resolution_text: str | None, source_count: int | None
) -> GuardrailResult:
    """Evaluate one ticket's outcome so far.

    category: the validated classification category.
    ticket_text: title + description, for prompt-injection detection only.
    resolution_text / source_count: the generated resolution's text and how many sources it
    cited, or (None, None) if no resolution was attempted (retrieval found no evidence).

    Precedence (highest first), the same style as routing's fixed category table:
    unsafe resolution text > security category > ungrounded > passed.
    """
    security = category == "security"
    injection = detect_prompt_injection(ticket_text)
    grounded = None if resolution_text is None else (source_count or 0) > 0
    blocked_reasons = unsafe_resolution_reasons(resolution_text) if resolution_text is not None else []

    if blocked_reasons:
        status = GuardrailStatus.BLOCKED_RESTRICTED_CONTENT
    elif security:
        status = GuardrailStatus.REQUIRES_HUMAN_REVIEW
    elif grounded is not True:  # covers both "no resolution attempted" and "cited nothing"
        status = GuardrailStatus.UNGROUNDED
    else:
        status = GuardrailStatus.PASSED
    return GuardrailResult(status, security, grounded, injection, blocked_reasons)
