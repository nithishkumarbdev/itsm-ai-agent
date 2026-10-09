import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.db.database import Base, engine
from app.main import app
from app.models import Ticket


def valid_payload(**overrides):
    payload = {
        "source": "web",
        "requester": "user@example.com",
        "title": "VPN not connecting",
        "description": "I cannot connect to the company VPN from home.",
    }
    payload.update(overrides)
    return payload


def count_tickets(db) -> int:
    return db.scalar(select(func.count()).select_from(Ticket))


# ---------- valid creation ----------


def test_create_ticket_returns_201_with_defaults(client):
    response = client.post("/tickets", json=valid_payload())

    assert response.status_code == 201
    body = response.json()
    uuid.UUID(body["id"])  # raises if the id is not a valid UUID
    assert body["source"] == "web"
    assert body["requester"] == "user@example.com"
    assert body["title"] == "VPN not connecting"
    assert body["description"] == "I cannot connect to the company VPN from home."
    assert body["status"] == "new"
    assert body["created_at"] is not None
    assert body["updated_at"] is not None


def test_ai_fields_are_empty_after_creation(client):
    """Phase 1 must not pretend classification or resolution exists."""
    body = client.post("/tickets", json=valid_payload()).json()

    for field in [
        "category",
        "priority",
        "urgency",
        "impact",
        "assigned_team",
        "classification_confidence",
        "retrieved_documents",
        "suggested_resolution",
        "resolution_confidence",
        "auto_resolution_eligible",
        "guardrail_status",
    ]:
        assert body[field] is None, f"{field} should be null before processing"


def test_each_ticket_gets_a_unique_id(client):
    first = client.post("/tickets", json=valid_payload()).json()
    second = client.post("/tickets", json=valid_payload()).json()

    assert first["id"] != second["id"]


def test_input_whitespace_is_stripped(client):
    response = client.post("/tickets", json=valid_payload(title="  VPN down  "))

    assert response.status_code == 201
    assert response.json()["title"] == "VPN down"


@pytest.mark.parametrize("source", ["web", "email", "slack", "other"])
def test_all_allowed_sources_are_accepted(client, source):
    response = client.post("/tickets", json=valid_payload(source=source))

    assert response.status_code == 201
    assert response.json()["source"] == source


# ---------- persistence ----------


def test_ticket_is_persisted_in_postgres(client, db):
    body = client.post("/tickets", json=valid_payload()).json()

    # `db` is a separate session from the one the request used.
    assert count_tickets(db) == 1
    stored = db.get(Ticket, uuid.UUID(body["id"]))

    assert stored is not None
    assert stored.source == "web"
    assert stored.requester == "user@example.com"
    assert stored.title == "VPN not connecting"
    assert stored.description == "I cannot connect to the company VPN from home."
    assert stored.status == "new"
    assert stored.category is None
    assert stored.auto_resolution_eligible is None
    assert stored.created_at is not None
    assert stored.updated_at is not None


# ---------- invalid requests ----------


def without(field):
    return {k: v for k, v in valid_payload().items() if k != field}


@pytest.mark.parametrize(
    "payload",
    [
        without("title"),
        without("description"),
        without("requester"),
        without("source"),
        valid_payload(requester="not-an-email"),
        valid_payload(source="carrier-pigeon"),
        valid_payload(title=""),
        valid_payload(title="   "),
        valid_payload(title="!!!"),
        valid_payload(title="x" * 201),
        valid_payload(description=""),
        valid_payload(description="---"),
        valid_payload(description="x" * 5001),
    ],
    ids=[
        "missing-title",
        "missing-description",
        "missing-requester",
        "missing-source",
        "bad-email",
        "bad-source",
        "empty-title",
        "whitespace-title",
        "symbols-only-title",
        "title-too-long",
        "empty-description",
        "symbols-only-description",
        "description-too-long",
    ],
)
def test_invalid_ticket_is_rejected_and_not_stored(client, db, payload):
    response = client.post("/tickets", json=payload)

    assert response.status_code == 422
    assert count_tickets(db) == 0


@pytest.mark.parametrize(
    "extra",
    [{"status": "resolved"}, {"category": "vpn"}, {"assigned_team": "Security"}],
    ids=["status", "category", "assigned_team"],
)
def test_client_cannot_set_backend_controlled_fields(client, db, extra):
    response = client.post("/tickets", json=valid_payload(**extra))

    assert response.status_code == 422
    assert count_tickets(db) == 0


# ---------- failure handling ----------


def test_database_unavailable_returns_503_without_leaking_details(unreachable_client):
    response = unreachable_client.post("/tickets", json=valid_payload())

    assert response.status_code == 503
    assert "id" not in response.json()  # no ticket was "created"
    # Generic message only: no host, port, driver or credentials.
    for leaked in ["127.0.0.1", "psycopg", "nobody", "nodb", "connection"]:
        assert leaked not in response.text


# ---------- startup ----------


def test_startup_creates_the_schema_on_a_fresh_database():
    """fresh database -> start app -> tickets table exists -> POST /tickets works."""
    Base.metadata.drop_all(bind=engine)  # simulate a brand-new database

    with TestClient(app) as started_client:  # `with` runs the startup hook
        response = started_client.post("/tickets", json=valid_payload())

    assert response.status_code == 201
