"""What happens when the database fails: nothing half-done is saved, and no internals leak."""

import uuid

import pytest
from sqlalchemy import func, select

from app.audit import record_event as real_record_event
from app.models import AuditEvent, AuditEventType, Ticket

SOME_ID = "00000000-0000-0000-0000-000000000000"


def orphan_event(db, ticket_id, event_type, message, details=None):
    """Stand-in for record_event that makes PostgreSQL reject the audit insert
    (foreign key to a ticket that does not exist)."""
    db.add(AuditEvent(ticket_id=uuid.uuid4(), event_type=event_type.value, message=message))


def count(db, model):
    return db.scalar(select(func.count()).select_from(model))


def assert_generic_500(response):
    assert response.status_code == 500
    assert response.json() == {"detail": "The request could not be completed due to an internal error."}
    for leaked in ["foreign key", "audit_events", "violates", "psycopg", "INSERT"]:
        assert leaked not in response.text


# ---------- atomicity: ticket change and audit event succeed or fail together ----------


def test_ticket_creation_is_rolled_back_if_the_audit_event_fails(client, db, monkeypatch):
    monkeypatch.setattr("app.api.routes.tickets.record_event", orphan_event)

    response = client.post(
        "/tickets",
        json={
            "source": "web",
            "requester": "user@example.com",
            "title": "VPN not connecting",
            "description": "I cannot connect to the company VPN from home.",
        },
    )

    assert_generic_500(response)
    assert count(db, Ticket) == 0  # no ticket without its audit event
    assert count(db, AuditEvent) == 0


def test_status_change_is_rolled_back_if_the_audit_event_fails(
    client, ticket, status_of, event_types_of, monkeypatch
):
    monkeypatch.setattr("app.lifecycle.record_event", orphan_event)

    response = client.post(f"/tickets/{ticket['id']}/status", json={"status": "processing"})

    assert_generic_500(response)
    assert status_of(ticket["id"]) == "new"  # the UPDATE was rolled back too
    assert event_types_of(ticket["id"]) == ["ticket_created"]


def test_process_is_rolled_back_if_the_second_audit_event_fails(
    client, ticket, status_of, event_types_of, monkeypatch
):
    # status_changed is written by the lifecycle; only the processing_started write fails.
    monkeypatch.setattr("app.workflow.record_event", orphan_event)

    response = client.post(f"/tickets/{ticket['id']}/process")

    assert_generic_500(response)
    assert status_of(ticket["id"]) == "new"
    assert event_types_of(ticket["id"]) == ["ticket_created"]  # status_changed rolled back too


@pytest.mark.parametrize(
    "failing_event",
    [AuditEventType.CLASSIFICATION_COMPLETED, AuditEventType.ROUTING_DECIDED],
    ids=["classification_completed-fails", "routing_decided-fails"],
)
def test_classification_and_routing_are_saved_together_or_not_at_all(
    client, ticket, row_of, event_types_of, monkeypatch, failing_event
):
    """Classification fields, team and both audit events commit in one transaction:
    if either audit write fails, none of it is saved."""

    def fail_only_for_one_event(db, ticket_id, event_type, message, details=None):
        if event_type == failing_event:
            orphan_event(db, ticket_id, event_type, message, details)
        else:
            real_record_event(db, ticket_id, event_type, message, details)

    monkeypatch.setattr("app.workflow.record_event", fail_only_for_one_event)

    response = client.post(f"/tickets/{ticket['id']}/process")

    assert_generic_500(response)
    row = row_of(ticket["id"])
    assert row.status == "processing"  # step 1 had already committed
    assert row.category is None and row.classification_confidence is None  # nothing partial saved
    assert row.assigned_team is None  # no team without its classification, and vice versa
    assert event_types_of(ticket["id"]) == ["ticket_created", "status_changed", "processing_started"]


# ---------- database unreachable ----------


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("GET", "/tickets", None),
        ("GET", f"/tickets/{SOME_ID}", None),
        ("POST", f"/tickets/{SOME_ID}/status", {"status": "processing"}),
        ("POST", f"/tickets/{SOME_ID}/process", None),
        ("GET", f"/tickets/{SOME_ID}/events", None),
        ("GET", "/knowledge", None),
        ("GET", f"/knowledge/{SOME_ID}", None),
        ("PATCH", f"/knowledge/{SOME_ID}", {"is_active": False}),
        ("POST", "/knowledge", {"title": "VPN Guide", "category": "vpn", "content": "## Symptoms\n- The VPN keeps dropping."}),
    ],
    ids=["list", "get", "status", "process", "events", "kb-list", "kb-get", "kb-patch", "kb-create"],
)
def test_unreachable_database_returns_503_without_leaking_details(unreachable_client, method, path, body):
    response = unreachable_client.request(method, path, json=body)

    assert response.status_code == 503
    assert response.json() == {"detail": "The database is currently unavailable. Try again later."}
    for leaked in ["127.0.0.1", "psycopg", "nobody", "nodb", "connection"]:
        assert leaked not in response.text
