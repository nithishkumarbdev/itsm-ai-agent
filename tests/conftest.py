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

import copy  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import uuid  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, select  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from app.api.deps import get_embedding_client, get_llm_client, get_retrieval_service  # noqa: E402
from app.db.database import Base, SessionLocal, engine, get_db  # noqa: E402
from app.main import app  # noqa: E402
from app.models import AuditEvent, KnowledgeDocument, Ticket  # noqa: E402
from app.services.indexing import index_document  # noqa: E402
from app.services.knowledge import seed_documents  # noqa: E402
from app.services.knowledge_seed import load_seed_documents  # noqa: E402
from app.services.retrieval import RetrievalService  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_schema():
    """Empty tables for every test, so tests are independent and repeatable."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


DEFAULT_CLASSIFICATION = {
    "category": "vpn",
    "priority": "P2",
    "urgency": "high",
    "impact": "individual",
    "confidence": 0.91,
    "reason": "The ticket describes a failure to connect to the corporate VPN.",
}


class MockLLMClient:
    """Stands in for the real LLM in every test. It is not part of the app on purpose,
    so production code can never end up using fake classifications.

    By default it returns a valid VPN classification. Tests change that with returns(),
    returns_raw() or fails_with(), and inspect what the service sent via `calls`.
    """

    name = "mock"

    def __init__(self):
        self.calls = []
        self.response = copy.deepcopy(DEFAULT_CLASSIFICATION)
        self.error = None
        self.before_return = None  # optional hook: runs while the "LLM call" is in progress

    def returns(self, **overrides):
        """Valid classification with some fields replaced."""
        self.response = {**DEFAULT_CLASSIFICATION, **overrides}

    def returns_raw(self, raw):
        """Exactly this output, however malformed."""
        self.response = raw

    def fails_with(self, error):
        self.error = error

    def generate_structured(self, *, system, user, schema_name, schema):
        self.calls.append(
            {"system": system, "user": user, "schema_name": schema_name, "schema": schema}
        )
        if self.before_return:
            self.before_return()
        if self.error:
            raise self.error
        return copy.deepcopy(self.response)


# Words that only describe asking for help, not the topic. A bag-of-words mock has no notion
# of meaning, so without this "I need help planning a vacation" would match IT guides that
# happen to contain "need" or "help".
_STOPWORDS = frozenset(
    """a an and are as at be been but by can do does for from has have how i if in into is it its
    my no not of on or our so that the their them then there these they this to up us was we were
    what when which who will with you your me am after before also any each few more most other
    some such than too very just need help please want best get got try thing something issue
    problem""".split()
)

# Similarity threshold for tests using the mock. It is specific to this mock: measured on the
# seed documents, relevant queries scored 0.28+ and irrelevant ones stayed below 0.15.
# A real embedding model has a different score scale (see RETRIEVAL_MIN_SCORE).
TEST_MIN_SCORE = 0.20


def _stem(word):
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


class MockEmbeddingClient:
    """Stands in for the real embedding model in every test. It is not part of the app.

    Deterministic bag-of-words hashing: each word (lower-cased, stemmed a little) is hashed to
    one of `dimension` slots, and the vector is normalised. So texts that share words get a high
    cosine similarity, and the same text always gives the same vector. It has no real semantic
    understanding: it exists to make retrieval tests stable, not to judge embedding quality.
    """

    def __init__(self, dimension=1024):
        self.dimension = dimension
        self.model = f"mock-bow-{dimension}"
        self.calls = []  # the list of texts of each embed_many call
        self.error = None
        self.before_return = None  # optional hook: runs while the "embedding call" is in progress

    def fails_with(self, error):
        self.error = error

    def embed_many(self, texts):
        self.calls.append(list(texts))
        if self.before_return:
            self.before_return()
        if self.error:
            raise self.error
        return [self._embed(text) for text in texts]

    def _embed(self, text):
        counts = {}
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in _STOPWORDS or len(word) < 2:
                continue
            word = _stem(word)
            counts[word] = counts.get(word, 0) + 1
        vector = [0.0] * self.dimension
        for word, count in counts.items():
            slot = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big") % self.dimension
            vector[slot] += 1.0 + math.log(count)
        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector] if norm else vector


@pytest.fixture()
def llm():
    return MockLLMClient()


@pytest.fixture()
def embedder():
    return MockEmbeddingClient()


@pytest.fixture()
def retriever(embedder):
    return RetrievalService(embedder, top_k=5, min_score=TEST_MIN_SCORE)


@pytest.fixture()
def client(llm, embedder, retriever):
    """Client that uses the app's real database dependency, pointed at the test database,
    and the mock LLM instead of a real provider.

    Not used as a context manager on purpose: the schema is managed by fresh_schema.
    (The startup hook itself is tested separately.)
    """
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_embedding_client] = lambda: embedder
    app.dependency_overrides[get_retrieval_service] = lambda: retriever
    yield TestClient(app)
    for dependency in (get_llm_client, get_embedding_client, get_retrieval_service):
        app.dependency_overrides.pop(dependency, None)


@pytest.fixture()
def db():
    """A separate session for looking inside the database directly from tests."""
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture()
def unreachable_client(llm, embedder, retriever):
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
    app.dependency_overrides[get_llm_client] = lambda: llm
    app.dependency_overrides[get_embedding_client] = lambda: embedder
    app.dependency_overrides[get_retrieval_service] = lambda: retriever
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


@pytest.fixture()
def row_of(db):
    """row_of(ticket_id): the Ticket row as stored in PostgreSQL right now (not via the API)."""

    def _row(ticket_id):
        db.expire_all()
        return db.get(Ticket, uuid.UUID(str(ticket_id)))

    return _row


@pytest.fixture()
def indexed_kb(db, embedder):
    """The 11 seed documents, stored and indexed with the mock embedder."""
    seed_documents(db, load_seed_documents())
    db.commit()
    for document in db.scalars(select(KnowledgeDocument).order_by(KnowledgeDocument.title)).all():
        index_document(db, document, embedder)
        db.commit()
    embedder.calls.clear()  # tests start counting embedding calls from zero
