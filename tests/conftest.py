import os

import pytest
from sqlalchemy.engine import make_url

# --- Safety checks. These run before the app is imported. ---------------------------
# The suite drops and recreates tables, so it must only ever run against a dedicated
# PostgreSQL test database, never the real one.
_test_url = os.environ.get("TEST_DATABASE_URL")
if not _test_url:
    pytest.exit(
        "TEST_DATABASE_URL is not set. Point it at a dedicated PostgreSQL database "
        "whose name ends in '_test' (see .env.example).",
        returncode=2,
    )
_parsed = make_url(_test_url)
if _parsed.get_backend_name() != "postgresql":
    pytest.exit("TEST_DATABASE_URL must be a PostgreSQL URL.", returncode=2)
if not (_parsed.database or "").endswith("_test"):
    pytest.exit(
        "Refusing to run: the TEST_DATABASE_URL database name must end in '_test'.",
        returncode=2,
    )

# Real environment variables beat .env, so the app itself now talks to the test database.
os.environ["DATABASE_URL"] = _test_url

import uuid  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.db.database import Base, SessionLocal, engine, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import AuditEvent, Ticket  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_schema():
    """Empty tables for every test, so tests are independent and repeatable."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client():
    """Client that uses the app's real database dependency, pointed at the test database.

    Not used as a context manager on purpose: the schema is managed by fresh_schema.
    (The startup hook itself is tested separately.)
    """
    return TestClient(app)


@pytest.fixture()
def db():
    """A separate session for looking inside the database directly from tests."""
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture()
def unreachable_client():
    """Client whose database is genuinely unreachable (nothing listens on port 1)."""
    dead_engine = create_engine(
        "postgresql+psycopg://nobody:nobody@127.0.0.1:1/nodb",
        connect_args={"connect_timeout": 2},
    )
    dead_sessions = sessionmaker(bind=dead_engine, autoflush=False, expire_on_commit=False)

    def dead_get_db():
        session = dead_sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = dead_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
    dead_engine.dispose()


@pytest.fixture()
def ticket(client):
    """A freshly created ticket (status 'new'), as returned by the API."""
    response = client.post(
        "/tickets",
        json={
            "source": "web",
            "requester": "user@example.com",
            "title": "VPN not connecting",
            "description": "I cannot connect to the company VPN from home.",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture()
def move_to(client):
    """move_to(ticket_id, "processing", "resolved") walks a ticket through legal statuses."""

    def _move(ticket_id, *statuses):
        for target in statuses:
            response = client.post(f"/tickets/{ticket_id}/status", json={"status": target})
            assert response.status_code == 200, response.text

    return _move


@pytest.fixture()
def status_of(db):
    """status_of(ticket_id): the status stored in PostgreSQL right now (not via the API)."""

    def _status_of(ticket_id):
        return db.scalar(select(Ticket.status).where(Ticket.id == uuid.UUID(str(ticket_id))))

    return _status_of


@pytest.fixture()
def event_types_of(db):
    """event_types_of(ticket_id): stored audit event types, oldest first (read from PostgreSQL)."""

    def _event_types_of(ticket_id):
        stmt = (
            select(AuditEvent.event_type)
            .where(AuditEvent.ticket_id == uuid.UUID(str(ticket_id)))
            .order_by(AuditEvent.id)
        )
        return list(db.scalars(stmt))

    return _event_types_of
