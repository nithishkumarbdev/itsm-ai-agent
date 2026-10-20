"""Indexing: knowledge documents -> chunks -> embeddings -> rows in PostgreSQL."""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.embeddings.base import EmbeddingError
from app.models import KnowledgeChunk, KnowledgeDocument
from app.services.indexing import embedding_text, index_document

CONTENT = (
    "## Symptoms\n- The VPN keeps dropping.\n\n"
    "## Troubleshooting Steps\n1. Reconnect the VPN client.\n\n"
    "## When to Escalate\nEscalate to the Network team."
)


def create_document(client, **overrides):
    body = {"title": "VPN Guide", "category": "vpn", "content": CONTENT}
    body.update(overrides)
    response = client.post("/knowledge", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def load(db, document_id):
    db.expire_all()
    return db.get(KnowledgeDocument, uuid.UUID(document_id))


def chunks_of(db, document_id):
    db.expire_all()
    return list(
        db.scalars(
            select(KnowledgeChunk)
            .where(KnowledgeChunk.document_id == uuid.UUID(document_id))
            .order_by(KnowledgeChunk.chunk_index)
        )
    )


def total_chunks(db):
    return db.scalar(select(func.count()).select_from(KnowledgeChunk))


# ---------- storing chunks and vectors ----------


def test_indexing_stores_one_chunk_per_section_with_its_vector(client, db, embedder):
    created = create_document(client)

    result = index_document(db, load(db, created["id"]), embedder)
    db.commit()

    assert (result.status, result.chunk_count, result.embedding_model) == ("indexed", 3, embedder.model)
    rows = chunks_of(db, created["id"])  # read back from PostgreSQL
    assert [r.chunk_index for r in rows] == [0, 1, 2]
    assert [r.section for r in rows] == ["Symptoms", "Troubleshooting Steps", "When to Escalate"]
    assert rows[0].content == "Symptoms\n- The VPN keeps dropping."
    for row in rows:
        assert row.document_id == uuid.UUID(created["id"])
        assert row.document_version == 1
        assert row.embedding_model == embedder.model
        assert len(row.embedding) == embedder.dimension  # the vector really is stored, full size
        assert row.created_at and row.updated_at


def test_the_stored_vector_is_the_embedding_of_the_title_plus_the_chunk(client, db, embedder):
    created = create_document(client)
    index_document(db, load(db, created["id"]), embedder)
    db.commit()

    row = chunks_of(db, created["id"])[1]

    expected = embedder.embed_many([embedding_text("VPN Guide", row.content)])[0]
    assert row.embedding == pytest.approx(expected)


def test_every_chunk_belongs_to_exactly_one_existing_document(client, db, embedder):
    created = create_document(client)
    index_document(db, load(db, created["id"]), embedder)
    db.commit()

    orphan = KnowledgeChunk(
        document_id=uuid.uuid4(), chunk_index=0, content="x", document_version=1,
        embedding_model=embedder.model, embedding=[0.0],
    )
    db.add(orphan)
    with pytest.raises(IntegrityError):  # the foreign key is enforced by PostgreSQL
        db.commit()
    db.rollback()


def test_the_database_refuses_two_chunks_with_the_same_position(client, db, embedder):
    created = create_document(client)
    index_document(db, load(db, created["id"]), embedder)
    db.commit()

    db.add(KnowledgeChunk(
        document_id=uuid.UUID(created["id"]), chunk_index=0, content="duplicate", document_version=1,
        embedding_model=embedder.model, embedding=[0.0],
    ))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()


# ---------- no duplicates ----------


def test_indexing_twice_changes_nothing_and_does_not_call_the_embedding_model_again(client, db, embedder):
    created = create_document(client)
    index_document(db, load(db, created["id"]), embedder)
    db.commit()
    ids_before = [r.id for r in chunks_of(db, created["id"])]
    calls_before = len(embedder.calls)

    second = index_document(db, load(db, created["id"]), embedder)
    db.commit()

    assert (second.status, second.chunk_count) == ("unchanged", 3)
    assert [r.id for r in chunks_of(db, created["id"])] == ids_before  # the very same rows
    assert total_chunks(db) == 3
    assert len(embedder.calls) == calls_before  # no embedding cost for an up-to-date document


# ---------- re-indexing ----------


def test_editing_a_document_and_reindexing_replaces_its_chunks(client, db, embedder):
    created = create_document(client)
    index_document(db, load(db, created["id"]), embedder)
    db.commit()
    old_ids = {r.id for r in chunks_of(db, created["id"])}
    new_content = "## Symptoms\n- The VPN cannot connect at all.\n\n## Troubleshooting Steps\n1. Restart the VPN client."
    assert client.patch(f"/knowledge/{created['id']}", json={"content": new_content}).json()["version"] == 2

    result = index_document(db, load(db, created["id"]), embedder)
    db.commit()

    rows = chunks_of(db, created["id"])
    assert (result.status, result.chunk_count) == ("indexed", 2)
    assert len(rows) == 2 and total_chunks(db) == 2  # no old chunks left behind
    assert old_ids.isdisjoint({r.id for r in rows})
    assert {r.document_version for r in rows} == {2}
    assert rows[0].content == "Symptoms\n- The VPN cannot connect at all."


def test_renaming_a_document_also_reindexes_it_because_the_title_is_embedded(client, db, embedder):
    created = create_document(client)
    index_document(db, load(db, created["id"]), embedder)
    db.commit()
    client.patch(f"/knowledge/{created['id']}", json={"title": "Remote Access Guide"})

    result = index_document(db, load(db, created["id"]), embedder)

    assert result.status == "indexed"


def test_deactivating_and_reactivating_needs_no_reindex(client, db, embedder):
    created = create_document(client)
    index_document(db, load(db, created["id"]), embedder)
    db.commit()
    client.patch(f"/knowledge/{created['id']}", json={"is_active": False})
    client.patch(f"/knowledge/{created['id']}", json={"is_active": True})  # version is unchanged

    result = index_document(db, load(db, created["id"]), embedder)

    assert result.status == "unchanged"


# ---------- inactive documents ----------


def test_an_inactive_document_is_not_indexed(client, db, embedder):
    created = create_document(client, is_active=False)

    result = index_document(db, load(db, created["id"]), embedder)
    db.commit()

    assert (result.status, result.chunk_count) == ("skipped_inactive", 0)
    assert total_chunks(db) == 0
    assert embedder.calls == []  # not even an embedding call


# ---------- failures leave the index as it was ----------


def test_embedding_failure_on_a_first_index_stores_nothing(client, db, embedder):
    created = create_document(client)
    embedder.fails_with(EmbeddingError("timeout"))

    with pytest.raises(EmbeddingError):
        index_document(db, load(db, created["id"]), embedder)
    db.rollback()

    assert total_chunks(db) == 0


def test_embedding_failure_on_a_reindex_keeps_the_old_chunks(client, db, embedder):
    created = create_document(client)
    index_document(db, load(db, created["id"]), embedder)
    db.commit()
    client.patch(f"/knowledge/{created['id']}", json={"content": "## Symptoms\n- Completely different text here."})
    embedder.fails_with(EmbeddingError("provider_error", "HTTP 500"))

    with pytest.raises(EmbeddingError):
        index_document(db, load(db, created["id"]), embedder)
    db.rollback()

    rows = chunks_of(db, created["id"])
    assert len(rows) == 3 and {r.document_version for r in rows} == {1}  # untouched


def test_vectors_of_the_wrong_size_from_a_provider_are_never_stored(client, db, embedder):
    created = create_document(client)
    embedder._embed = lambda text: [1.0, 0.0]  # a "provider" that returns 2 numbers, not 1024

    with pytest.raises(EmbeddingError) as excinfo:
        index_document(db, load(db, created["id"]), embedder)
    db.rollback()

    assert excinfo.value.kind == "invalid_response"
    assert total_chunks(db) == 0


# ---------- POST /knowledge/{id}/index ----------


def test_index_endpoint_indexes_and_repeating_it_is_harmless(client, db):
    created = create_document(client)

    first = client.post(f"/knowledge/{created['id']}/index")
    second = client.post(f"/knowledge/{created['id']}/index")

    assert first.status_code == 200
    assert first.json() == {
        "document_id": created["id"], "title": "VPN Guide", "status": "indexed",
        "chunk_count": 3, "embedding_model": "mock-bow-1024",
    }
    assert second.json()["status"] == "unchanged"
    assert total_chunks(db) == 3


def test_index_endpoint_reports_skipped_inactive_documents(client):
    created = create_document(client, is_active=False)

    assert client.post(f"/knowledge/{created['id']}/index").json()["status"] == "skipped_inactive"


def test_index_endpoint_for_an_unknown_document_returns_404(client):
    assert client.post(f"/knowledge/{uuid.uuid4()}/index").status_code == 404


def test_index_endpoint_returns_503_without_details_when_embedding_fails(client, db, embedder):
    created = create_document(client)
    embedder.fails_with(EmbeddingError("provider_error", "HTTP 401 for secret-key-123"))

    response = client.post(f"/knowledge/{created['id']}/index")

    assert response.status_code == 503
    assert response.json() == {"detail": "The embedding service is currently unavailable. Try again later."}
    assert "secret-key-123" not in response.text
    assert total_chunks(db) == 0
