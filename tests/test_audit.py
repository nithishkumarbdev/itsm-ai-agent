import uuid
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.models import AuditEvent


def test_creating_a_ticket_records_a_ticket_created_event(client, db, ticket):
    response = client.get(f"/tickets/{ticket['id']}/events")

    assert response.status_code == 200
    events = response.json()
    assert len(events) == 1
    assert events[0]["event_type"] == "ticket_created"
    assert events[0]["message"] == "Ticket created"
    assert events[0]["ticket_id"] == ticket["id"]
    assert events[0]["details"] == {"source": "web"}
    assert events[0]["created_at"] is not None

    # ...and it is really stored in PostgreSQL, not just returned by the API.
    stored = db.scalars(select(AuditEvent).where(AuditEvent.ticket_id == uuid.UUID(ticket["id"]))).all()
    assert [e.event_type for e in stored] == ["ticket_created"]


def test_events_for_unknown_ticket_returns_404(client):
    response = client.get(f"/tickets/{uuid.uuid4()}/events")

    assert response.status_code == 404


def test_events_are_returned_in_chronological_order(client, ticket):
    ticket_id = ticket["id"]
    client.post(f"/tickets/{ticket_id}/process")
    client.post(f"/tickets/{ticket_id}/status", json={"status": "awaiting_human"})
    client.post(f"/tickets/{ticket_id}/status", json={"status": "processing"})

    events = client.get(f"/tickets/{ticket_id}/events").json()

    assert [e["event_type"] for e in events] == [
        "ticket_created",
        "status_changed",
        "processing_started",
        "status_changed",
        "status_changed",
    ]
    status_changes = [e["details"] for e in events if e["event_type"] == "status_changed"]
    assert status_changes == [
        {"from_status": "new", "to_status": "processing"},
        {"from_status": "processing", "to_status": "awaiting_human"},
        {"from_status": "awaiting_human", "to_status": "processing"},
    ]
    timestamps = [datetime.fromisoformat(e["created_at"]) for e in events]
    assert timestamps == sorted(timestamps)
    ids = [e["id"] for e in events]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)


def test_audit_event_must_belong_to_an_existing_ticket(db):
    """The foreign key is enforced by PostgreSQL itself."""
    db.add(AuditEvent(ticket_id=uuid.uuid4(), event_type="ticket_created", message="orphan"))

    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


def test_audit_trail_is_append_only_through_the_api(client, ticket):
    """There are no endpoints to edit or delete tickets or audit events."""
    ticket_id = ticket["id"]

    assert client.delete(f"/tickets/{ticket_id}").status_code == 405
    assert client.put(f"/tickets/{ticket_id}/events").status_code == 405
    assert client.delete(f"/tickets/{ticket_id}/events").status_code == 405
    assert client.post(f"/tickets/{ticket_id}/events").status_code == 405
    assert client.get("/audit-events").status_code == 404
