"""Retrieval: similarity search over indexed chunks, checked against PostgreSQL.

Part 1 uses hand-made vectors, so ranking, the threshold, the category bonus, staleness and
tie-breaking are tested exactly, independent of any embedding model.
Part 2 indexes the real seed documents with the mock embedder and searches them.
"""

import math
import uuid

import pytest
from sqlalchemy import select

from app.api.deps import get_retrieval_service
from app.embeddings.base import EmbeddingError, embed_one
from app.main import app
from app.models import KnowledgeChunk, KnowledgeDocument
from app.services.retrieval import CATEGORY_BOOST, KnowledgeIndexError, RetrievalService, cosine_similarity

QUERY = [1.0, 0.0]


def at_similarity(similarity):
    """A unit vector whose cosine similarity with QUERY is `similarity`."""
    return [similarity, math.sqrt(1 - similarity**2)]


def make_document(client, title, category="vpn", **extra):
    body = {"title": title, "category": category, "content": f"## Symptoms\n- Text of the {title}.", **extra}
    response = client.post("/knowledge", json=body)
    assert response.status_code == 201, response.text
    return response.json()


def add_chunk(db, document, vector, *, index=0, version=None, model="mock-bow-1024", content=None):
    db.add(KnowledgeChunk(
        document_id=uuid.UUID(document["id"]), chunk_index=index, section="Symptoms",
        content=content or f"chunk {index} of {document['title']}",
        document_version=document["version"] if version is None else version,
        embedding_model=model, embedding=vector,
    ))
    db.commit()


def titles(outcome):
    return [r.document_title for r in outcome.results]


# =====================================================================================
# Part 1: exact behaviour with hand-made vectors
# =====================================================================================


def test_results_are_ordered_best_first_with_their_similarity(client, db, retriever):
    for title, sim in [("Low", 0.3), ("High", 0.9), ("Mid", 0.6)]:
        add_chunk(db, make_document(client, title), at_similarity(sim))

    outcome = retriever.search(db, QUERY)

    assert titles(outcome) == ["High", "Mid", "Low"]
    assert [r.similarity_score for r in outcome.results] == pytest.approx([0.9, 0.6, 0.3], abs=1e-4)
    assert outcome.candidates_considered == 3


def test_top_k_limits_the_number_of_results(client, db, retriever):
    for i in range(10):
        add_chunk(db, make_document(client, f"Doc {i:02d}"), at_similarity(0.9 - i * 0.05))

    assert titles(retriever.search(db, QUERY, top_k=3)) == ["Doc 00", "Doc 01", "Doc 02"]
    assert len(retriever.search(db, QUERY, top_k=100).results) == 10  # fewer qualify than asked for
    assert len(retriever.search(db, QUERY).results) == 5  # the service default


def test_top_k_below_one_is_rejected(db, retriever):
    with pytest.raises(ValueError):
        retriever.search(db, QUERY, top_k=0)


def test_chunks_below_the_threshold_are_never_returned(client, db, retriever):
    for title, sim in [("A", 0.9), ("B", 0.6), ("C", 0.3), ("D", 0.1)]:
        add_chunk(db, make_document(client, title), at_similarity(sim))

    outcome = retriever.search(db, QUERY, min_score=0.25)

    assert titles(outcome) == ["A", "B", "C"]  # D (0.1) is excluded
    assert all(r.similarity_score >= 0.25 for r in outcome.results)
    assert outcome.candidates_considered == 4  # it was considered, just not relevant enough


def test_when_nothing_passes_the_threshold_the_result_is_an_explicit_empty_list(client, db, retriever):
    add_chunk(db, make_document(client, "A"), at_similarity(0.5))

    outcome = retriever.search(db, QUERY, min_score=0.95)

    assert outcome.results == []  # never "the closest thing"
    assert outcome.candidates_considered == 1


def test_an_empty_index_gives_an_empty_result_and_zero_candidates(db, retriever):
    outcome = retriever.search(db, QUERY)

    assert outcome.results == [] and outcome.candidates_considered == 0


def test_a_chunk_of_the_wanted_category_is_ranked_first_but_its_score_is_not_inflated(client, db, retriever):
    add_chunk(db, make_document(client, "VPN Guide", "vpn"), at_similarity(0.60))
    add_chunk(db, make_document(client, "Network Guide", "network"), at_similarity(0.65))

    plain = retriever.search(db, QUERY)
    boosted = retriever.search(db, QUERY, category="vpn")

    assert titles(plain) == ["Network Guide", "VPN Guide"]  # by similarity alone
    assert titles(boosted) == ["VPN Guide", "Network Guide"]  # 0.60 + bonus beats 0.65
    assert CATEGORY_BOOST > 0.05
    scores = {r.document_title: r.similarity_score for r in boosted.results}
    assert scores == pytest.approx({"VPN Guide": 0.60, "Network Guide": 0.65}, abs=1e-4)  # raw, unchanged
    assert [r.category_match for r in boosted.results] == [True, False]
    assert not any(r.category_match for r in plain.results)


def test_the_category_bonus_cannot_make_an_irrelevant_chunk_pass_the_threshold(client, db, retriever):
    add_chunk(db, make_document(client, "VPN Guide", "vpn"), at_similarity(0.05))  # 0.05 alone; 0.15 with the bonus
    add_chunk(db, make_document(client, "Other Guide", "hardware"), at_similarity(0.90))

    outcome = retriever.search(db, QUERY, category="vpn", min_score=0.10)

    assert titles(outcome) == ["Other Guide"]  # the vpn chunk is below 0.10 on its own similarity


def test_other_categories_are_still_found_when_a_category_is_given(client, db, retriever):
    add_chunk(db, make_document(client, "MFA Guide", "account"), at_similarity(0.8))

    assert titles(retriever.search(db, QUERY, category="vpn")) == ["MFA Guide"]  # not filtered out


def test_equal_scores_are_ordered_deterministically_by_title_then_position(client, db, retriever):
    vector = at_similarity(0.7)
    add_chunk(db, make_document(client, "B Guide"), vector)
    add_chunk(db, make_document(client, "A Guide"), vector)

    first = retriever.search(db, QUERY)
    second = retriever.search(db, QUERY)

    assert titles(first) == ["A Guide", "B Guide"]
    assert first == second


def test_inactive_documents_are_never_retrieved_and_come_back_when_reactivated(client, db, retriever, embedder):
    document = make_document(client, "VPN Guide")
    add_chunk(db, document, at_similarity(0.9))
    calls_before = len(embedder.calls)

    client.patch(f"/knowledge/{document['id']}", json={"is_active": False})
    assert retriever.search(db, QUERY).results == []
    assert retriever.search(db, QUERY).candidates_considered == 0  # not even a candidate

    client.patch(f"/knowledge/{document['id']}", json={"is_active": True})
    assert titles(retriever.search(db, QUERY)) == ["VPN Guide"]  # no re-indexing needed
    assert len(embedder.calls) == calls_before


def test_chunks_of_an_older_document_version_are_never_retrieved(client, db, retriever):
    document = make_document(client, "VPN Guide")
    add_chunk(db, document, at_similarity(0.9))  # built from version 1

    client.patch(f"/knowledge/{document['id']}", json={"content": "## Symptoms\n- Rewritten text of the guide."})

    outcome = retriever.search(db, QUERY)  # the document is now version 2, but not re-indexed
    assert outcome.results == [] and outcome.candidates_considered == 0


def test_stale_and_current_chunks_are_never_returned_together(client, db, retriever):
    document = make_document(client, "VPN Guide")
    add_chunk(db, document, at_similarity(0.9), index=0, version=1, content="OLD text")
    client.patch(f"/knowledge/{document['id']}", json={"content": "## Symptoms\n- Rewritten text of the guide."})
    add_chunk(db, {**document, "version": 2}, at_similarity(0.8), index=1, version=2, content="NEW text")

    outcome = retriever.search(db, QUERY)

    assert [r.content for r in outcome.results] == ["NEW text"]
    assert {r.version for r in outcome.results} == {2}


def test_an_index_built_with_another_embedding_model_is_refused_not_searched(client, db, retriever):
    add_chunk(db, make_document(client, "VPN Guide"), at_similarity(0.9), model="some-other-model")

    with pytest.raises(KnowledgeIndexError, match="re-run indexing"):
        retriever.search(db, QUERY)


def test_vectors_of_a_different_size_are_refused_not_compared(client, db, retriever):
    add_chunk(db, make_document(client, "VPN Guide"), [1.0, 0.0, 0.0])  # 3 numbers, query has 2

    with pytest.raises(KnowledgeIndexError):
        retriever.search(db, QUERY)


# =====================================================================================
# Part 2: the real seed documents, indexed with the mock embedder
# =====================================================================================

FIVE_SECTIONS = {"Symptoms", "Common Causes", "Troubleshooting Steps", "Restricted Actions", "When to Escalate"}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("VPN keeps disconnecting after I connect", "VPN Troubleshooting Guide"),
        ("My office printer is not printing", "Printer Troubleshooting Guide"),
        ("I am not receiving my MFA approval", "MFA Troubleshooting Guide"),
        ("My account is locked after several login attempts", "Account Lockout Guide"),
        ("Wi-Fi keeps dropping in the office", "Wi-Fi Troubleshooting Guide"),
        ("Teams meeting audio not working", "Microsoft Teams Troubleshooting Guide"),
        ("laptop is very slow and the fan is loud", "Laptop Performance Troubleshooting Guide"),
        ("I forgot my password", "Password Reset Guide"),
        ("emails not syncing on my phone", "Email Sync Troubleshooting Guide"),
        ("I need to install new software", "Software Installation Guide"),
        ("I think someone accessed my company account", "Suspected Account Compromise Guide"),
    ],
)
def test_the_right_guide_is_the_top_result(db, indexed_kb, retriever, query, expected):
    outcome = retriever.retrieve(db, query)

    assert outcome.results, "expected at least one relevant chunk"
    assert outcome.results[0].document_title == expected


@pytest.mark.parametrize(
    "query",
    [
        "I need help planning a vacation",
        "What is the best pizza recipe for a birthday party",
        "Who won the football match yesterday",
        "How do I train for a marathon",
    ],
)
def test_unrelated_questions_get_no_knowledge_instead_of_a_random_guide(db, indexed_kb, retriever, query):
    outcome = retriever.retrieve(db, query)

    assert outcome.results == []
    assert outcome.candidates_considered == 55  # the whole index was searched; nothing was relevant


def test_each_result_can_be_traced_back_to_its_source(db, indexed_kb, retriever):
    result = retriever.retrieve(db, "VPN keeps disconnecting after I connect").results[0]

    document = db.get(KnowledgeDocument, result.document_id)
    chunk = db.get(KnowledgeChunk, result.chunk_id)
    assert document.title == result.document_title == "VPN Troubleshooting Guide"
    assert document.category == result.category == "vpn"
    assert document.version == result.version == 1
    assert chunk.document_id == document.id
    assert chunk.chunk_index == result.chunk_index
    assert chunk.content == result.content
    assert result.section in FIVE_SECTIONS
    assert 0.0 <= result.similarity_score <= 1.0
    assert result.category_match is False  # no category was given


def test_results_without_a_category_are_sorted_by_similarity(db, indexed_kb, retriever):
    outcome = retriever.retrieve(db, "steps to troubleshoot and when to escalate", top_k=20, min_score=0.0)

    scores = [r.similarity_score for r in outcome.results]
    assert len(scores) == 20
    assert scores == sorted(scores, reverse=True)


def test_search_is_repeatable(db, indexed_kb, retriever):
    query = "VPN keeps disconnecting after I connect"

    assert retriever.retrieve(db, query) == retriever.retrieve(db, query)


def test_top_k_on_the_real_index(db, indexed_kb, retriever):
    assert len(retriever.retrieve(db, "troubleshooting steps", top_k=3, min_score=0.0).results) == 3
    assert len(retriever.retrieve(db, "troubleshooting steps", top_k=100, min_score=0.0).results) == 55


def test_results_match_an_independent_brute_force_calculation(db, indexed_kb, retriever, embedder):
    query = "my VPN connection drops and I cannot reach the intranet"
    query_vector = embed_one(embedder, query)
    rows = db.execute(select(KnowledgeChunk, KnowledgeDocument).join(KnowledgeDocument)).all()
    expected = sorted(
        ((cosine_similarity(query_vector, c.embedding), d.title, c.chunk_index) for c, d in rows), key=lambda t: (-t[0], t[1], t[2])
    )
    expected = [t for t in expected if t[0] >= retriever.min_score][: retriever.top_k]

    outcome = retriever.retrieve(db, query)

    assert [(r.document_title, r.chunk_index) for r in outcome.results] == [(t, i) for _, t, i in expected]
    assert [r.similarity_score for r in outcome.results] == pytest.approx([s for s, _, _ in expected], abs=1e-4)


def test_threshold_boundary_on_the_real_index(db, indexed_kb, retriever, embedder):
    query = "printer is not printing"
    top = retriever.retrieve(db, query, top_k=1).results[0]
    chunk = db.get(KnowledgeChunk, top.chunk_id)
    exact = cosine_similarity(embed_one(embedder, query), chunk.embedding)

    assert top.chunk_id in {r.chunk_id for r in retriever.retrieve(db, query, min_score=exact).results}
    assert top.chunk_id not in {r.chunk_id for r in retriever.retrieve(db, query, min_score=exact + 1e-9).results}


def test_a_given_category_ranks_first_but_does_not_hide_other_documents(db, indexed_kb, retriever):
    query = "I am not receiving my MFA approval"

    with_category = retriever.retrieve(db, query, category="vpn")
    without = retriever.retrieve(db, query)

    assert "MFA Troubleshooting Guide" in titles(with_category)  # still found although the category is vpn
    raw = lambda outcome: {r.chunk_id: r.similarity_score for r in outcome.results}  # noqa: E731
    shared = raw(with_category).keys() & raw(without).keys()
    assert shared and all(raw(with_category)[c] == raw(without)[c] for c in shared)  # scores never inflated
    assert all(r.category == "vpn" for r in with_category.results if r.category_match)


def test_deactivated_guides_disappear_from_search_and_return_without_reindexing(client, db, indexed_kb, retriever, embedder):
    vpn = next(d for d in client.get("/knowledge", params={"category": "vpn"}).json())
    query = "VPN keeps disconnecting after I connect"

    client.patch(f"/knowledge/{vpn['id']}", json={"is_active": False})
    assert "VPN Troubleshooting Guide" not in titles(retriever.retrieve(db, query))

    client.patch(f"/knowledge/{vpn['id']}", json={"is_active": True})
    assert titles(retriever.retrieve(db, query))[0] == "VPN Troubleshooting Guide"
    assert embedder.calls == [[query], [query]]  # only the two queries were embedded: nothing was re-indexed


def test_an_edited_guide_is_not_searchable_until_it_is_reindexed_and_never_twice(client, db, indexed_kb, retriever):
    printer = client.get("/knowledge", params={"category": "hardware"}).json()
    printer = next(d for d in printer if d["title"] == "Printer Troubleshooting Guide")
    query = "My office printer is not printing"
    assert titles(retriever.retrieve(db, query))[0] == "Printer Troubleshooting Guide"

    new_text = "## Symptoms\n- The office printer does not print anything.\n\n## Troubleshooting Steps\n1. Check the printer queue."
    client.patch(f"/knowledge/{printer['id']}", json={"content": new_text})
    assert "Printer Troubleshooting Guide" not in titles(retriever.retrieve(db, query))  # old chunks are ignored

    assert client.post(f"/knowledge/{printer['id']}/index").json()["status"] == "indexed"
    outcome = retriever.retrieve(db, query)
    printer_chunks = [r for r in outcome.results if r.document_title == "Printer Troubleshooting Guide"]
    assert printer_chunks and {r.version for r in printer_chunks} == {2}
    assert all("Restricted Actions" != r.section for r in printer_chunks)  # only the new 2 sections exist
    assert len({r.chunk_id for r in printer_chunks}) == len(printer_chunks)


# =====================================================================================
# The search API
# =====================================================================================


def search(client, **body):
    return client.post("/knowledge/search", json=body)


def test_search_endpoint_returns_results_with_scores_and_sources(client, indexed_kb, llm):
    response = search(client, query="VPN keeps disconnecting after I connect", top_k=3, category="vpn")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"results", "candidates_considered", "top_k", "min_score"}
    assert (body["top_k"], body["min_score"], body["candidates_considered"]) == (3, 0.2, 55)
    assert 1 <= len(body["results"]) <= 3
    first = body["results"][0]
    assert set(first) == {"document_id", "document_title", "category", "version", "chunk_id", "chunk_index",
                          "section", "content", "similarity_score", "category_match"}
    assert first["document_title"] == "VPN Troubleshooting Guide" and first["category_match"] is True
    assert llm.calls == []  # searching never involves the LLM


def test_search_endpoint_can_override_the_threshold_for_tuning(client, indexed_kb):
    assert search(client, query="VPN keeps disconnecting", min_score=0.99).json()["results"] == []
    assert len(search(client, query="VPN keeps disconnecting", min_score=0.0, top_k=20).json()["results"]) == 20


def test_search_endpoint_says_so_when_nothing_is_relevant(client, indexed_kb):
    body = search(client, query="I need help planning a vacation").json()

    assert body["results"] == [] and body["candidates_considered"] == 55


def test_search_endpoint_on_an_empty_index_shows_zero_candidates(client):
    body = search(client, query="VPN keeps disconnecting").json()

    assert body["results"] == [] and body["candidates_considered"] == 0


@pytest.mark.parametrize(
    "body",
    [
        {}, {"query": ""}, {"query": "   "}, {"query": "!!!"}, {"query": "x" * 2001},
        {"query": "vpn", "top_k": 0}, {"query": "vpn", "top_k": 21}, {"query": "vpn", "top_k": "many"},
        {"query": "vpn", "min_score": -0.1}, {"query": "vpn", "min_score": 1.5},
        {"query": "vpn", "category": "banana"}, {"query": "vpn", "unknown": 1},
    ],
    ids=["no-query", "empty", "blank", "symbols", "too-long", "top_k-0", "top_k-21", "top_k-text",
         "min_score-low", "min_score-high", "bad-category", "unknown-field"],
)
def test_invalid_search_requests_are_rejected(client, indexed_kb, body):
    assert client.post("/knowledge/search", json=body).status_code == 422


def test_search_returns_503_without_details_when_the_query_cannot_be_embedded(client, indexed_kb, embedder):
    embedder.fails_with(EmbeddingError("provider_error", "HTTP 401 for secret-key-123"))

    response = search(client, query="VPN keeps disconnecting")

    assert response.status_code == 503
    assert response.json() == {"detail": "The embedding service is currently unavailable. Try again later."}
    assert "secret-key-123" not in response.text


def test_search_returns_503_when_the_index_was_built_with_another_model(client, indexed_kb, embedder):
    other = RetrievalService(type(embedder)(dimension=256), top_k=5, min_score=0.2)
    app.dependency_overrides[get_retrieval_service] = lambda: other  # the fixture removes it afterwards

    response = search(client, query="VPN keeps disconnecting")

    assert response.status_code == 503
    assert "must be rebuilt" in response.json()["detail"]
