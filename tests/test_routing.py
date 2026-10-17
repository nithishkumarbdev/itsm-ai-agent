"""Routing on its own: a pure function of the category. No LLM, no database involved."""

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from app.schemas.classification import Category
from app.services import routing
from app.services.routing import CATEGORY_TO_TEAM, FALLBACK_TEAM, Team, route_ticket

# Written out by hand, independent of the implementation. Changing the policy in
# app/services/routing.py means changing it here on purpose.
EXPECTED_TEAMS = {
    "vpn": "Network",
    "network": "Network",
    "security": "Security",
    "hardware": "Hardware / IT Support",
    "access": "Access Management",
    "account": "Access Management",
    "email": "IT Support",
    "software": "IT Support",
    "other": "IT Support",
}


@pytest.mark.parametrize(("category", "team"), list(EXPECTED_TEAMS.items()), ids=list(EXPECTED_TEAMS))
def test_category_routes_to_the_expected_team(category, team):
    decision = route_ticket(category)

    assert decision.assigned_team == team
    assert decision.assigned_team.value == team
    assert decision.rule == f"category:{category}"


def test_every_category_has_a_routing_rule():
    """Adding a category without deciding where it goes must fail a test."""
    assert set(CATEGORY_TO_TEAM) == set(Category)
    assert set(EXPECTED_TEAMS) == {c.value for c in Category}


def test_team_names_are_exactly_the_specified_ones():
    assert {t.value for t in Team} == {
        "Security",
        "Network",
        "Hardware / IT Support",
        "Access Management",
        "IT Support",
    }


def test_category_enum_members_are_accepted_too():
    assert route_ticket(Category.VPN) == route_ticket("vpn")


# ---------- security precedence ----------


def test_security_is_the_only_category_routed_to_security_and_nothing_overrides_it():
    assert [c for c, team in CATEGORY_TO_TEAM.items() if team is Team.SECURITY] == [Category.SECURITY]
    decision = route_ticket("security")
    assert decision.assigned_team is Team.SECURITY
    assert decision.rule == "category:security"


def test_routing_depends_on_the_category_and_nothing_else():
    """No priority, confidence or ticket text can change the team: the only input is the category."""
    assert list(inspect.signature(route_ticket).parameters) == ["category"]


# ---------- missing / invalid category ----------


@pytest.mark.parametrize(
    "bad",
    [None, "", " ", "banana", "VPN", "Security", "SECURITY", " vpn", "vpn ", 42, ["vpn"]],
    ids=["none", "empty", "space", "junk", "upper-vpn", "capital-security", "upper-security",
         "leading-space", "trailing-space", "int", "list"],
)
def test_missing_or_invalid_category_gets_the_safe_fallback(bad):
    decision = route_ticket(bad)

    assert decision.assigned_team is FALLBACK_TEAM is Team.IT_SUPPORT
    assert decision.rule == "fallback:no_valid_category"
    assert decision.reason == "Fallback routing because the classification category is missing or invalid"


def test_categories_are_not_case_normalised():
    """Same rule as the classification schema: exactly lower case. No guessing what "VPN" meant."""
    assert route_ticket("VPN").rule == "fallback:no_valid_category"
    assert route_ticket("vpn").rule == "category:vpn"


# ---------- determinism and explanation ----------


def test_same_input_always_gives_the_same_decision():
    for category in [*EXPECTED_TEAMS, None, "banana"]:
        assert len({route_ticket(category) for _ in range(50)}) == 1


def test_reason_is_a_short_fixed_sentence():
    assert route_ticket("vpn").reason == "vpn category routed to Network team"
    assert route_ticket("hardware").reason == "hardware category routed to Hardware / IT Support team"
    assert route_ticket("security").reason == "security category routed to Security team"


def test_decision_cannot_be_modified_after_it_is_made():
    decision = route_ticket("vpn")

    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.assigned_team = Team.SECURITY


# ---------- independence from the LLM ----------


def test_routing_module_has_no_dependency_on_the_llm_database_or_network():
    tree = ast.parse(Path(routing.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)

    forbidden = ("app.llm", "app.services.classification", "sqlalchemy", "anthropic", "httpx", "requests")
    assert not [m for m in imported if m and m.startswith(forbidden)], imported
