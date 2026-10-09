EXPECTED_ENDPOINTS = {
    ("/health", "get"),
    ("/tickets", "get"),
    ("/tickets", "post"),
    ("/tickets/{ticket_id}", "get"),
    ("/tickets/{ticket_id}/status", "post"),
    ("/tickets/{ticket_id}/process", "post"),
    ("/tickets/{ticket_id}/events", "get"),
}


def test_docs_and_openapi_are_served(client):
    assert client.get("/docs").status_code == 200

    response = client.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()

    # Exactly these endpoints: nothing extra, in particular no PUT/PATCH/DELETE.
    actual = {(path, method) for path, operations in spec["paths"].items() for method in operations}
    assert actual == EXPECTED_ENDPOINTS
    assert {"TicketCreate", "TicketRead", "StatusUpdate", "AuditEventRead"} <= set(
        spec["components"]["schemas"]
    )
