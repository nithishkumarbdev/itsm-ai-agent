def test_health_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_503_when_database_is_unreachable(unreachable_client):
    response = unreachable_client.get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "unavailable"}
