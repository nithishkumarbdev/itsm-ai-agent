import uuid
from datetime import datetime

from sqlalchemy import select

from app.models import Ticket


def create(client, title):
    response = client.post(
        "/tickets",
        json={
            "source": "web",
            "requester": "user@example.com",
            "title": title,
            "description": f"Description for {title}",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_get_ticket_returns_the_stored_record(client, db, ticket):
    response = client.get(f"/tickets/{ticket['id']}")

    assert response.status_code == 200
    body = response.json()
    stored = db.scalars(select(Ticket).where(Ticket.id == uuid.UUID(ticket["id"]))).one()
    assert body["id"] == str(stored.id)
    assert body["source"] == stored.source
    assert body["requester"] == stored.requester
    assert body["title"] == stored.title
    assert body["description"] == stored.description
    assert body["status"] == stored.status == "new"
    assert datetime.fromisoformat(body["created_at"]) == stored.created_at
    assert datetime.fromisoformat(body["updated_at"]) == stored.updated_at


def test_get_unknown_ticket_returns_404(client):
    response = client.get(f"/tickets/{uuid.uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Ticket not found"}


def test_get_ticket_with_malformed_id_returns_422(client):
    assert client.get("/tickets/not-a-uuid").status_code == 422


def test_list_is_empty_when_there_are_no_tickets(client):
    response = client.get("/tickets")

    assert response.status_code == 200
    assert response.json() == []


def test_list_returns_created_tickets_newest_first(client):
    first = create(client, "First ticket")
    second = create(client, "Second ticket")
    third = create(client, "Third ticket")

    response = client.get("/tickets")

    assert response.status_code == 200
    assert [t["id"] for t in response.json()] == [third["id"], second["id"], first["id"]]


def test_list_pagination(client):
    first = create(client, "First ticket")
    second = create(client, "Second ticket")
    third = create(client, "Third ticket")

    page_one = client.get("/tickets", params={"limit": 2}).json()
    page_two = client.get("/tickets", params={"limit": 2, "offset": 2}).json()

    assert [t["id"] for t in page_one] == [third["id"], second["id"]]
    assert [t["id"] for t in page_two] == [first["id"]]


def test_list_rejects_out_of_range_pagination(client):
    assert client.get("/tickets", params={"limit": 0}).status_code == 422
    assert client.get("/tickets", params={"limit": 201}).status_code == 422
    assert client.get("/tickets", params={"offset": -1}).status_code == 422
