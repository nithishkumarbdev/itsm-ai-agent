"""The /knowledge endpoints, checked against PostgreSQL."""

import uuid
from datetime import datetime

import pytest
from sqlalchemy import func, select

from app.models import KnowledgeDocument

CONTENT = "## Symptoms\n- The VPN client keeps disconnecting.\n\n## Troubleshooting Steps\n1. Disconnect and reconnect."
METADATA = {
    "department": "IT Support",
    "source": "internal_kb",
    "document_type": "troubleshooting",
    "audience": "employees",
}


def payload(**overrides):
    body = {"title": "VPN Troubleshooting Guide", "category": "vpn", "content": CONTENT, "metadata": METADATA}
    body.update(overrides)
    return body


def without(field):
    return {k: v for k, v in payload().items() if k != field}


def create(client, **overrides):
    response = client.post("/knowledge", json=payload(**overrides))
    assert response.status_code == 201, response.text
    return response.json()


def patch(client, document_id, **body):
    return client.patch(f"/knowledge/{document_id}", json=body)


def count(db):
    return db.scalar(select(func.count()).select_from(KnowledgeDocument))


def stored(db, document_id):
    db.expire_all()
    return db.get(KnowledgeDocument, uuid.UUID(document_id))


def when(text):
    return datetime.fromisoformat(text)


# ---------- create ----------


def test_create_document_returns_201_and_is_stored(client, db):
    response = client.post("/knowledge", json=payload())

    assert response.status_code == 201
    body = response.json()
    uuid.UUID(body["id"])
    assert body["title"] == "VPN Troubleshooting Guide"
    assert body["category"] == "vpn"
    assert body["content"] == CONTENT
    assert body["version"] == 1
    assert body["is_active"] is True
    assert body["metadata"] == METADATA
    assert body["created_at"] and body["updated_at"]

    row = stored(db, body["id"])  # what PostgreSQL holds
    assert (row.title, row.category, row.content, row.version, row.is_active) == (
        "VPN Troubleshooting Guide", "vpn", CONTENT, 1, True,
    )
    assert row.doc_metadata == METADATA
    assert count(db) == 1


def test_create_uses_defaults_for_version_active_and_metadata(client):
    body = client.post("/knowledge", json=without("metadata")).json()

    assert body["version"] == 1
    assert body["is_active"] is True
    assert body["metadata"] == {}


def test_create_accepts_an_explicit_version(client):
    assert create(client, version=3)["version"] == 3


def test_response_does_not_leak_internals(client):
    body = create(client)

    assert set(body) == {"id", "title", "category", "content", "version", "is_active", "metadata", "created_at", "updated_at"}


INVALID = {
    "missing-title": without("title"),
    "empty-title": payload(title=""),
    "whitespace-title": payload(title="   "),
    "symbols-only-title": payload(title="!!!"),
    "title-too-long": payload(title="x" * 201),
    "missing-category": without("category"),
    "unknown-category": payload(category="banana"),
    "uppercase-category": payload(category="VPN"),
    "missing-content": without("content"),
    "empty-content": payload(content=""),
    "whitespace-content": payload(content="   \n  "),
    "too-short-content": payload(content="x"),
    "symbols-only-content": payload(content="-" * 40),
    "content-too-long": payload(content="x" * 50_001),
    "version-zero": payload(version=0),
    "version-negative": payload(version=-1),
    "version-fraction": payload(version=1.5),
    "version-text": payload(version="v2"),
    "version-null": payload(version=None),
    "metadata-list": payload(metadata=["a"]),
    "metadata-string": payload(metadata="x"),
    "metadata-null": payload(metadata=None),
    "is-active-junk": payload(is_active="maybe"),
    "unknown-field": payload(id=str(uuid.uuid4())),
}


@pytest.mark.parametrize("body", list(INVALID.values()), ids=list(INVALID))
def test_invalid_document_is_rejected_and_not_stored(client, db, body):
    response = client.post("/knowledge", json=body)

    assert response.status_code == 422
    assert count(db) == 0


def test_duplicate_title_is_rejected_with_409(client, db):
    create(client)

    response = client.post("/knowledge", json=payload(content=CONTENT + "\n2. Another step."))

    assert response.status_code == 409
    assert "VPN Troubleshooting Guide" in response.json()["detail"]
    assert count(db) == 1


# ---------- read ----------


def test_get_document_returns_the_full_stored_document(client, db):
    created = create(client)

    response = client.get(f"/knowledge/{created['id']}")

    assert response.status_code == 200
    body = response.json()
    row = stored(db, created["id"])
    assert body["id"] == str(row.id)
    assert body["content"] == row.content == CONTENT
    assert body["metadata"] == row.doc_metadata
    assert when(body["created_at"]) == row.created_at


def test_get_unknown_document_returns_404(client):
    response = client.get(f"/knowledge/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Knowledge document not found"}


def test_get_with_malformed_id_returns_422(client):
    assert client.get("/knowledge/not-a-uuid").status_code == 422


def test_list_is_empty_without_documents(client):
    response = client.get("/knowledge")

    assert response.status_code == 200
    assert response.json() == []


def test_list_returns_summaries_without_content_ordered_by_title(client):
    create(client, title="B Guide")
    create(client, title="A Guide")

    documents = client.get("/knowledge").json()

    assert [d["title"] for d in documents] == ["A Guide", "B Guide"]
    for d in documents:
        assert "content" not in d
        assert set(d) == {"id", "title", "category", "version", "is_active", "metadata", "created_at", "updated_at"}


def test_list_filters_by_category(client):
    create(client, title="VPN Guide", category="vpn")
    create(client, title="Printer Guide", category="hardware")

    assert [d["title"] for d in client.get("/knowledge", params={"category": "hardware"}).json()] == ["Printer Guide"]
    assert client.get("/knowledge", params={"category": "email"}).json() == []
    assert client.get("/knowledge", params={"category": "banana"}).status_code == 422


def test_list_pagination(client):
    for name in ["A Guide", "B Guide", "C Guide"]:
        create(client, title=name)

    first = client.get("/knowledge", params={"limit": 2}).json()
    rest = client.get("/knowledge", params={"limit": 2, "offset": 2}).json()

    assert [d["title"] for d in first] == ["A Guide", "B Guide"]
    assert [d["title"] for d in rest] == ["C Guide"]
    assert client.get("/knowledge", params={"limit": 0}).status_code == 422
    assert client.get("/knowledge", params={"offset": -1}).status_code == 422


# ---------- active and inactive documents ----------


def test_deactivating_keeps_the_document_stored_and_readable(client, db):
    created = create(client)

    response = patch(client, created["id"], is_active=False)

    assert response.status_code == 200
    assert response.json()["is_active"] is False
    assert stored(db, created["id"]).is_active is False  # still in the database
    assert client.get(f"/knowledge/{created['id']}").json()["is_active"] is False  # still readable by id


def test_list_can_show_only_active_or_only_inactive_documents(client):
    active = create(client, title="Active Guide")
    inactive = create(client, title="Inactive Guide")
    patch(client, inactive["id"], is_active=False)

    only_active = client.get("/knowledge", params={"is_active": "true"}).json()
    only_inactive = client.get("/knowledge", params={"is_active": "false"}).json()
    everything = client.get("/knowledge").json()

    assert [d["id"] for d in only_active] == [active["id"]]
    assert [d["id"] for d in only_inactive] == [inactive["id"]]
    assert len(everything) == 2


def test_a_document_can_be_reactivated(client):
    created = create(client)
    patch(client, created["id"], is_active=False)

    response = patch(client, created["id"], is_active=True)

    assert response.json()["is_active"] is True


def test_a_document_can_be_created_inactive(client):
    assert create(client, is_active=False)["is_active"] is False


# ---------- update and version behaviour ----------


def test_changing_content_updates_the_document_and_raises_the_version(client, db):
    created = create(client)

    response = patch(client, created["id"], content="## Symptoms\n- Updated symptoms for the VPN guide.")

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "## Symptoms\n- Updated symptoms for the VPN guide."
    assert body["version"] == 2
    assert body["created_at"] == created["created_at"]  # set once
    assert when(body["updated_at"]) > when(created["updated_at"])
    row = stored(db, created["id"])
    assert (row.version, row.content) == (2, body["content"])


@pytest.mark.parametrize(
    "change",
    [{"title": "VPN Guide (renamed)"}, {"category": "network"}, {"metadata": {"department": "Network"}}],
    ids=["title", "category", "metadata"],
)
def test_each_content_field_raises_the_version_when_it_changes(client, change):
    created = create(client)

    body = patch(client, created["id"], **change).json()

    assert body["version"] == 2
    for key, value in change.items():
        assert body[key] == value


def test_every_change_raises_the_version_by_one(client):
    created = create(client)

    patch(client, created["id"], content="## Symptoms\n- Second version of the guide.")
    body = patch(client, created["id"], content="## Symptoms\n- Third version of the guide.").json()

    assert body["version"] == 3


def test_deactivating_alone_does_not_change_the_version(client):
    created = create(client)

    body = patch(client, created["id"], is_active=False).json()

    assert body["version"] == 1
    assert when(body["updated_at"]) > when(created["updated_at"])


def test_sending_values_that_are_already_stored_changes_nothing(client):
    created = create(client)

    response = patch(client, created["id"], title=created["title"], content=created["content"], metadata=METADATA)

    assert response.status_code == 200
    assert response.json()["version"] == 1
    assert response.json()["updated_at"] == created["updated_at"]


def test_content_change_and_deactivation_in_one_request_raise_the_version_once(client):
    created = create(client)

    body = patch(client, created["id"], content="## Symptoms\n- Replaced content, retired document.", is_active=False).json()

    assert body["version"] == 2
    assert body["is_active"] is False


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"title": None},
        {"is_active": None},
        {"metadata": None},
        {"version": 5},
        {"id": str(uuid.uuid4())},
        {"category": "banana"},
        {"title": ""},
        {"title": "   "},
        {"content": ""},
        {"content": "too short"},
        {"metadata": ["x"]},
        {"is_active": "maybe"},
        {"unknown": 1},
    ],
    ids=["empty", "null-title", "null-active", "null-metadata", "version", "id", "bad-category", "empty-title",
         "blank-title", "empty-content", "short-content", "metadata-list", "active-junk", "unknown-field"],
)
def test_invalid_update_is_rejected_and_changes_nothing(client, db, body):
    created = create(client)

    response = client.patch(f"/knowledge/{created['id']}", json=body)

    assert response.status_code == 422
    row = stored(db, created["id"])
    assert (row.title, row.category, row.content, row.version, row.is_active) == (
        created["title"], created["category"], created["content"], 1, True,
    )


def test_renaming_to_an_existing_title_is_rejected(client, db):
    create(client, title="A Guide")
    other = create(client, title="B Guide")

    response = patch(client, other["id"], title="A Guide")

    assert response.status_code == 409
    row = stored(db, other["id"])
    assert (row.title, row.version) == ("B Guide", 1)


def test_updating_an_unknown_document_returns_404(client):
    assert patch(client, uuid.uuid4(), is_active=False).status_code == 404


def test_updating_with_a_malformed_id_returns_422(client):
    assert patch(client, "not-a-uuid", is_active=False).status_code == 422


# ---------- documents are never deleted ----------


def test_there_is_no_way_to_delete_or_replace_a_document(client, db):
    created = create(client)

    assert client.delete(f"/knowledge/{created['id']}").status_code == 405
    assert client.put(f"/knowledge/{created['id']}", json=payload()).status_code == 405
    assert count(db) == 1


# ---------- ticket processing only retrieves from what has been indexed ----------


def test_ticket_processing_retrieves_only_from_indexed_documents(client, llm):
    create(client)  # a VPN guide exists, but it has not been indexed
    ticket = client.post(
        "/tickets",
        json={"source": "web", "requester": "user@example.com", "title": "VPN not connecting",
              "description": "I cannot connect to the company VPN from home."},
    ).json()

    body = client.post(f"/tickets/{ticket['id']}/process").json()

    events = [e["event_type"] for e in client.get(f"/tickets/{ticket['id']}/events").json()]
    assert events == ["ticket_created", "status_changed", "processing_started", "classification_completed",
                      "routing_decided", "knowledge_retrieved", "guardrails_checked"]
    assert body["assigned_team"] == "Network"
    assert body["retrieved_documents"] == []  # searched, but nothing is indexed yet
    for field in ["suggested_resolution", "resolution_confidence", "auto_resolution_eligible"]:
        assert body[field] is None
    assert len(llm.calls) == 1  # classification only
    assert "Symptoms" not in llm.calls[0]["user"] + llm.calls[0]["system"]  # no knowledge text reaches the model
