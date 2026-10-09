from datetime import datetime

import pytest

from app.models import TicketStatus


def invalid(current, target):
    return f"Invalid ticket status transition: {current} -> {target}"


# ---------- POST /tickets/{id}/status: legal moves ----------


def test_valid_transition_updates_ticket_and_records_event(client, ticket, status_of, event_types_of):
    response = client.post(f"/tickets/{ticket['id']}/status", json={"status": "processing"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processing"
    assert status_of(ticket["id"]) == "processing"  # stored in PostgreSQL
    assert body["created_at"] == ticket["created_at"]  # set once, never changed
    assert datetime.fromisoformat(body["updated_at"]) > datetime.fromisoformat(ticket["updated_at"])

    assert event_types_of(ticket["id"]) == ["ticket_created", "status_changed"]
    event = client.get(f"/tickets/{ticket['id']}/events").json()[-1]
    assert event["message"] == "Ticket moved from new to processing"
    assert event["details"] == {"from_status": "new", "to_status": "processing"}


@pytest.mark.parametrize(
    "path",
    [
        ["processing", "resolved", "closed"],
        ["processing", "awaiting_human", "escalated", "closed"],
        ["processing", "escalated", "processing", "awaiting_human", "processing", "resolved"],
    ],
    ids=["resolved-then-closed", "human-then-escalated-then-closed", "loops-back-to-processing"],
)
def test_legal_paths_through_the_lifecycle(client, ticket, move_to, status_of, path):
    move_to(ticket["id"], *path)

    assert status_of(ticket["id"]) == path[-1]


# ---------- POST /tickets/{id}/status: illegal moves ----------


@pytest.mark.parametrize(
    ("setup", "target"),
    [
        ([], "resolved"),
        ([], "closed"),
        ([], "awaiting_human"),
        ([], "escalated"),
        (["processing"], "new"),
        (["processing"], "processing"),
        (["processing", "resolved"], "escalated"),
        (["processing", "resolved"], "processing"),
        (["processing", "escalated"], "resolved"),
    ],
    ids=[
        "new-to-resolved",
        "new-to-closed",
        "new-to-awaiting_human",
        "new-to-escalated",
        "processing-to-new",
        "processing-to-processing",
        "resolved-to-escalated",
        "resolved-to-processing",
        "escalated-to-resolved",
    ],
)
def test_illegal_transition_is_rejected_and_changes_nothing(
    client, ticket, move_to, status_of, event_types_of, setup, target
):
    move_to(ticket["id"], *setup)
    current = setup[-1] if setup else "new"
    events_before = event_types_of(ticket["id"])

    response = client.post(f"/tickets/{ticket['id']}/status", json={"status": target})

    assert response.status_code == 409
    assert response.json() == {"detail": invalid(current, target)}
    assert status_of(ticket["id"]) == current
    assert event_types_of(ticket["id"]) == events_before  # no misleading status_changed event


@pytest.mark.parametrize("target", [s.value for s in TicketStatus])
def test_closed_ticket_is_terminal(client, ticket, move_to, status_of, event_types_of, target):
    move_to(ticket["id"], "processing", "resolved", "closed")
    events_before = event_types_of(ticket["id"])

    response = client.post(f"/tickets/{ticket['id']}/status", json={"status": target})

    assert response.status_code == 409
    assert status_of(ticket["id"]) == "closed"
    assert event_types_of(ticket["id"]) == events_before


# ---------- POST /tickets/{id}/status: bad requests ----------


@pytest.mark.parametrize(
    "body",
    [{"status": "banana"}, {}, {"status": "processing", "category": "vpn"}],
    ids=["unknown-status", "missing-status", "extra-field"],
)
def test_malformed_status_request_returns_422(client, ticket, status_of, body):
    response = client.post(f"/tickets/{ticket['id']}/status", json=body)

    assert response.status_code == 422
    assert status_of(ticket["id"]) == "new"


def test_status_change_for_unknown_ticket_returns_404(client):
    response = client.post(
        "/tickets/00000000-0000-0000-0000-000000000000/status", json={"status": "processing"}
    )

    assert response.status_code == 404


# ---------- POST /tickets/{id}/process ----------


def test_process_moves_ticket_to_processing_and_records_events(
    client, ticket, status_of, event_types_of
):
    response = client.post(f"/tickets/{ticket['id']}/process")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processing"
    assert status_of(ticket["id"]) == "processing"
    assert event_types_of(ticket["id"]) == ["ticket_created", "status_changed", "processing_started"]
    started = client.get(f"/tickets/{ticket['id']}/events").json()[-1]
    assert started["message"] == "Ticket processing started"


def test_process_does_not_fake_any_ai_output(client, ticket):
    body = client.post(f"/tickets/{ticket['id']}/process").json()

    for field in [
        "category",
        "priority",
        "assigned_team",
        "classification_confidence",
        "retrieved_documents",
        "suggested_resolution",
        "resolution_confidence",
        "auto_resolution_eligible",
        "guardrail_status",
    ]:
        assert body[field] is None


@pytest.mark.parametrize(
    "setup",
    [["processing", "awaiting_human"], ["processing", "escalated"]],
    ids=["from-awaiting_human", "from-escalated"],
)
def test_process_can_restart_processing_from_human_states(client, ticket, move_to, status_of, setup):
    move_to(ticket["id"], *setup)

    response = client.post(f"/tickets/{ticket['id']}/process")

    assert response.status_code == 200
    assert status_of(ticket["id"]) == "processing"


@pytest.mark.parametrize(
    "setup",
    [
        ["processing"],
        ["processing", "resolved"],
        ["processing", "resolved", "closed"],
    ],
    ids=["already-processing", "resolved", "closed"],
)
def test_process_is_rejected_when_processing_must_not_start(
    client, ticket, move_to, status_of, event_types_of, setup
):
    move_to(ticket["id"], *setup)
    current = setup[-1]
    events_before = event_types_of(ticket["id"])

    response = client.post(f"/tickets/{ticket['id']}/process")

    assert response.status_code == 409
    assert response.json() == {"detail": invalid(current, "processing")}
    assert status_of(ticket["id"]) == current
    assert event_types_of(ticket["id"]) == events_before  # no processing_started event


def test_process_unknown_ticket_returns_404(client):
    response = client.post("/tickets/00000000-0000-0000-0000-000000000000/process")

    assert response.status_code == 404
