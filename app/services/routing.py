"""Deterministic routing: which team gets a ticket.

A pure function of the validated category. No LLM, no database, no network, and nothing
else (priority, confidence, ticket text) can change the result. The same input always
gives the same team, so routing is predictable and easy to audit.
"""

from dataclasses import dataclass
from enum import StrEnum

from app.schemas.classification import Category


class Team(StrEnum):
    SECURITY = "Security"
    NETWORK = "Network"
    HARDWARE = "Hardware / IT Support"
    ACCESS_MANAGEMENT = "Access Management"
    IT_SUPPORT = "IT Support"


# The whole routing policy. Each category maps to exactly one team, so no rule can
# override another: a security ticket always goes to Security.
CATEGORY_TO_TEAM: dict[Category, Team] = {
    Category.SECURITY: Team.SECURITY,
    Category.VPN: Team.NETWORK,
    Category.NETWORK: Team.NETWORK,
    Category.HARDWARE: Team.HARDWARE,
    Category.ACCESS: Team.ACCESS_MANAGEMENT,
    Category.ACCOUNT: Team.ACCESS_MANAGEMENT,
    Category.EMAIL: Team.IT_SUPPORT,
    Category.SOFTWARE: Team.IT_SUPPORT,
    Category.OTHER: Team.IT_SUPPORT,
}

# Used when there is no valid category. It is the general service desk: never a
# guess at a specialist team, and never Security by accident.
FALLBACK_TEAM = Team.IT_SUPPORT


@dataclass(frozen=True)
class RoutingDecision:
    assigned_team: Team
    rule: str  # which rule fired, e.g. "category:vpn"
    reason: str  # short, fixed-format explanation for the audit trail


def route_ticket(category: str | Category | None) -> RoutingDecision:
    """Return the team for a category.

    The category must be exactly one of the known values (lower case). Anything else
    (None, "", "VPN", " vpn", junk) is not reinterpreted or guessed: it gets the fallback.
    """
    try:
        valid = Category(category)
    except (ValueError, TypeError):
        return RoutingDecision(
            assigned_team=FALLBACK_TEAM,
            rule="fallback:no_valid_category",
            reason="Fallback routing because the classification category is missing or invalid",
        )
    team = CATEGORY_TO_TEAM[valid]
    return RoutingDecision(
        assigned_team=team,
        rule=f"category:{valid.value}",
        reason=f"{valid.value} category routed to {team.value} team",
    )
