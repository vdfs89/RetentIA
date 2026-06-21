from fastapi.testclient import TestClient

from src.main import app


def test_metrics_endpoint():
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text
