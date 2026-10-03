from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint_checks_service_name():
    client = TestClient(app)
    response = client.get("/health")
    assert response.json()["service"] == "financial-doc-fraud-detection"
