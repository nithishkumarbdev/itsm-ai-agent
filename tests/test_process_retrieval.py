"""POST /tickets/{id}/process with retrieval: classify -> route -> retrieve -> store -> audit."""

import json
import uuid

import pytest
from sqlalchemy import select, update
from sqlalchemy.exc import OperationalError

from app.api.deps import get_retrieval_service
from app.db.database import SessionLocal
from app.embeddings.base import EmbeddingError
from app.llm.base import LLMError
from app.main import app
from app.models import Ticket
from app.services.retrieval import RetrievalService

VPN_TITLE = "VPN keeps disconnecting"
VPN_DESCRIPTION = "My VPN connects but disconnects every few minutes when I am working remotely."
REFERENCE_KEYS = {"document_id", "document_title", "category", "version", "chunk_id", "chunk_index", "similarity_score"}


def create(client, title=VPN_TITLE, description=VPN_DESCRIPTION):
    response = client.post(
        "/tickets",
        json={"source": "web", "requester": "user@example.com", "title": title, "description": description},
    )
    assert response.status_code == 201, response.text
    return response.json()


def process(client, ticket_id):
    response = client.post(f"/tickets/{ticket_id}/process")
    assert response.status_code == 200, response.text
    return response.json()


def events_of(client, ticket_id):
    return client.get(f"/tickets/{ticket_id}/events").json()


def types_of(events):
    return [e["event_type"] for e in events]


# ---------- the demo scenario ----------


def test_vpn_ticket_is_classified_routed_and_matched_to_the_vpn_guide(client, indexed_kb, llm, embedder, row_of):
    ticket = create()

    body = process(client, ticket["id"])

    assert body["category"] == "vpn"
    assert body["assigned_team"] == "Network"
    assert body["status"] == "processing"  # not resolved
    references = body["retrieved_documents"]
    assert references[0]["document_title"] == "VPN Troubleshooting Guide"
    for reference in references:
        assert set(reference) == REFERENCE_KEYS  # ids, source and score: not the text
        assert reference["similarity_score"] >= 0.2
    assert row_of(ticket["id"]).retrieved_documents == references  # stored in PostgreSQL

    events = events_of(client, ticket["id"])
    assert types_of(events) == [
        "ticket_created", "status_changed", "processing_started",
        "classification_completed", "routing_decided", "knowledge_retrieved",
    ]
    assert events[-1]["details"] == {
        "result_count": len(references),
        "top_score": max(r["similarity_score"] for r in references),
        "category": "vpn",
        "min_score": 0.2,
        "candidates_considered": 55,
        "documents": list(dict.fromkeys(r["document_title"] for r in references)),
    }
    assert events[-1]["message"] == f"Retrieved {len(references)} knowledge chunk(s)"


def test_only_the_problem_is_searched_for_and_the_model_never_sees_the_knowledge(client, indexed_kb, llm, embedder):
    ticket = create()

    process(client, ticket["id"])

    assert embedder.calls == [[f"{VPN_TITLE}\n{VPN_DESCRIPTION}"]]  # title and description, nothing else
    assert ticket["id"] not in embedder.calls[0][0]
    assert len(llm.calls) == 1  # classification only: no second LLM call to "use" the evidence
    assert "Symptoms" not in llm.calls[0]["user"] + llm.calls[0]["system"]


def test_no_resolution_is_generated_and_nothing_is_automated(client, indexed_kb):
    body = process(client, create()["id"])

    for field in ["suggested_resolution", "resolution_confidence", "auto_resolution_eligible", "guardrail_status"]:
        assert body[field] is None
    assert body["status"] == "processing"


def test_a_security_ticket_gets_evidence_but_is_only_routed_never_resolved(client, indexed_kb, llm):
    llm.returns(category="security", priority="P1", reason="Possible unauthorised access to an account.")
    ticket = create("Suspicious account access", "I think someone accessed my company account.")

    body = process(client, ticket["id"])

    assert body["category"] == "security"
    assert body["assigned_team"] == "Security"
    assert body["retrieved_documents"][0]["document_title"] == "Suspected Account Compromise Guide"
    assert body["status"] == "processing"
    assert body["suggested_resolution"] is None and body["auto_resolution_eligible"] is None


def test_retrieval_supplies_evidence_but_never_changes_the_routing(client, indexed_kb, llm):
    llm.returns(category="vpn")  # the classification says vpn ...
    ticket = create("Printer not printing", "My office printer is not printing.")  # ... the text is about printers

    body = process(client, ticket["id"])

    assert "Printer Troubleshooting Guide" in [r["document_title"] for r in body["retrieved_documents"]]
    assert body["assigned_team"] == "Network"  # routing still comes only from the category


# ---------- nothing relevant ----------


def test_no_relevant_knowledge_is_recorded_as_an_empty_list_not_a_random_guide(client, indexed_kb, llm, row_of):
    llm.returns(category="other", priority="P4", urgency="low", reason="Not an IT problem.")
    ticket = create("Vacation planning", "I need help planning a vacation.")

    body = process(client, ticket["id"])

    assert body["retrieved_documents"] == []
    assert row_of(ticket["id"]).retrieved_documents == []  # [] means "searched, nothing relevant" (null means "not searched")
    event = events_of(client, ticket["id"])[-1]
    assert event["event_type"] == "knowledge_retrieved"
    assert event["message"] == "No sufficiently relevant knowledge found"
    assert event["details"]["result_count"] == 0 and event["details"]["top_score"] is None
    assert body["status"] == "processing"  # no escalation policy yet


def test_an_empty_index_is_visible_as_zero_candidates(client, llm):
    body = process(client, create()["id"])

    assert body["retrieved_documents"] == []
    assert events_of(client, body["id"])[-1]["details"]["candidates_considered"] == 0


# ---------- failures ----------


@pytest.mark.parametrize(
    "error",
    [EmbeddingError("timeout"), EmbeddingError("provider_error", "HTTP 500"), EmbeddingError("invalid_response", "bad")],
    ids=["timeout", "provider-error", "invalid-response"],
)
def test_retrieval_failure_keeps_classification_and_routing_and_hands_over_to_a_person(
    client, indexed_kb, embedder, row_of, error
):
    embedder.fails_with(error)
    ticket = create()

    body = process(client, ticket["id"])

    assert body["status"] == "awaiting_human"  # never resolved
    assert (body["category"], body["assigned_team"]) == ("vpn", "Network")  # still valid, still saved
    assert body["retrieved_documents"] is None  # explicitly "not retrieved"
    row = row_of(ticket["id"])
    assert (row.category, row.assigned_team, row.retrieved_documents) == ("vpn", "Network", None)

    events = events_of(client, ticket["id"])
    assert types_of(events) == [
        "ticket_created", "status_changed", "processing_started",
        "classification_completed", "routing_decided", "retrieval_failed", "status_changed",
    ]
    assert events[5]["details"]["error_type"] == error.kind
    assert "knowledge_retrieved" not in types_of(events)  # no fake retrieval


def test_an_index_from_another_embedding_model_is_reported_not_searched(client, indexed_kb, embedder):
    other = RetrievalService(type(embedder)(dimension=256), top_k=5, min_score=0.2)
    app.dependency_overrides[get_retrieval_service] = lambda: other  # the fixture removes it afterwards
    ticket = create()

    body = process(client, ticket["id"])

    assert body["status"] == "awaiting_human"
    assert (body["category"], body["assigned_team"]) == ("vpn", "Network")
    assert body["retrieved_documents"] is None
    failed = next(e for e in events_of(client, ticket["id"]) if e["event_type"] == "retrieval_failed")
    assert failed["details"] == {"error_type": "index_incompatible"}


def test_retrieval_is_not_attempted_when_classification_fails(client, indexed_kb, llm, embedder):
    llm.fails_with(LLMError("timeout"))
    ticket = create()

    body = process(client, ticket["id"])

    assert body["status"] == "awaiting_human" and body["assigned_team"] is None
    assert embedder.calls == []  # the embedding model was never called
    assert not {"knowledge_retrieved", "retrieval_failed"} & set(types_of(events_of(client, ticket["id"])))


def test_retrieval_is_not_attempted_when_routing_fails(client, indexed_kb, embedder, monkeypatch):
    def broken_router(category):
        raise RuntimeError("internal routing problem")

    monkeypatch.setattr("app.workflow.route_ticket", broken_router)

    body = process(client, create()["id"])

    assert body["status"] == "awaiting_human"
    assert embedder.calls == []


# ---------- reprocessing ----------


def test_reprocessing_replaces_the_evidence_instead_of_adding_to_it(client, indexed_kb, move_to):
    ticket = create()
    first = process(client, ticket["id"])["retrieved_documents"]
    move_to(ticket["id"], "awaiting_human")

    second = process(client, ticket["id"])["retrieved_documents"]

    assert second == first  # same evidence, not doubled
    assert types_of(events_of(client, ticket["id"])).count("knowledge_retrieved") == 2  # both runs are in the history


def test_a_failed_reprocess_does_not_leave_old_evidence_next_to_a_new_classification(
    client, indexed_kb, embedder, move_to, row_of
):
    ticket = create()
    assert process(client, ticket["id"])["retrieved_documents"]  # first run found evidence
    move_to(ticket["id"], "awaiting_human")
    embedder.fails_with(EmbeddingError("timeout"))

    body = process(client, ticket["id"])

    assert body["status"] == "awaiting_human"
    assert body["retrieved_documents"] is None
    assert row_of(ticket["id"]).retrieved_documents is None


# ---------- transactions ----------


def test_no_database_lock_is_held_while_the_query_is_being_embedded(client, indexed_kb, embedder):
    ticket = create()
    observed = []

    def during_embedding_call():
        session = SessionLocal()
        try:
            session.execute(
                select(Ticket.id).where(Ticket.id == uuid.UUID(ticket["id"])).with_for_update(nowait=True)
            )
            observed.append("row was free")
        except OperationalError:
            observed.append("row was locked")
        finally:
            session.rollback()
            session.close()

    embedder.before_return = during_embedding_call
    process(client, ticket["id"])

    assert observed == ["row was free"]


def test_result_is_discarded_if_the_ticket_changed_while_the_query_was_being_embedded(
    client, indexed_kb, embedder, row_of
):
    ticket = create()

    def someone_resolves_the_ticket_meanwhile():
        session = SessionLocal()
        session.execute(update(Ticket).where(Ticket.id == uuid.UUID(ticket["id"])).values(status="resolved"))
        session.commit()
        session.close()

    embedder.before_return = someone_resolves_the_ticket_meanwhile
    body = process(client, ticket["id"])

    assert body["status"] == "resolved"  # their change stands
    assert body["category"] is None and body["retrieved_documents"] is None
    assert not {"classification_completed", "knowledge_retrieved"} & set(types_of(events_of(client, ticket["id"])))
