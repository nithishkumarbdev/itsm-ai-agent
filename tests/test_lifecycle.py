from itertools import product

import pytest

from app.lifecycle import ALLOWED_TRANSITIONS, InvalidTransition, can_transition
from app.models import TicketStatus

# Written out by hand, independent of the implementation. If someone changes the
# lifecycle in app/lifecycle.py, this test forces them to change it here on purpose.
EXPECTED_ALLOWED = {
    ("new", "processing"),
    ("processing", "awaiting_human"),
    ("processing", "resolved"),
    ("processing", "escalated"),
    ("awaiting_human", "processing"),
    ("awaiting_human", "resolved"),
    ("awaiting_human", "escalated"),
    ("escalated", "processing"),
    ("escalated", "closed"),
    ("resolved", "closed"),
}


def test_every_status_has_a_transition_entry():
    assert set(ALLOWED_TRANSITIONS) == set(TicketStatus)


@pytest.mark.parametrize(
    ("current", "target"),
    list(product(TicketStatus, TicketStatus)),
    ids=lambda status: status.value,
)
def test_transition_table(current, target):
    """All 36 (current, target) pairs: allowed exactly when listed above."""
    assert can_transition(current, target) == ((current.value, target.value) in EXPECTED_ALLOWED)


def test_closed_is_terminal():
    assert ALLOWED_TRANSITIONS[TicketStatus.CLOSED] == frozenset()


def test_invalid_transition_message():
    error = InvalidTransition(TicketStatus.CLOSED, TicketStatus.PROCESSING)

    assert str(error) == "Invalid ticket status transition: closed -> processing"
