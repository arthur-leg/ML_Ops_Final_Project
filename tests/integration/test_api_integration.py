"""Integration test for the Flask API using Flask's built-in test client.

Unlike the E2E test (which hits a real running deployment over HTTP),
this exercises the full request -> validate_row -> predict_row -> response
flow in-process, using the placeholder prediction path (no MLflow/model
required). This is fast enough to run in the "PR -> dev" CI stage, before
any model has been trained or promoted.

Run: pytest tests/integration/test_api_integration.py -v
"""
import pytest

from backend.api import app

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.get_json()["status"] == "ok"

    def test_metrics_endpoint_exposes_prediction_metrics(self, client):
        response = client.get("/metrics")

        assert response.status_code == 200
        body = response.data.decode("utf-8")
        assert "prediction_requests_total" in body
        assert "prediction_request_latency_seconds" in body
        assert "prediction_failed_requests_total" in body
        assert "backend_health_status" in body


class TestPredictEndpointIntegration:
    def test_valid_payload_returns_hpi_prediction(self, client, auth_headers):
        payload = {
            "country": "FR",
            "year": 2022,
            "hicp": 105.0,
            "unemployment_rate": 7.5,
        }
        response = client.post("/predict", json=payload, headers=auth_headers)

        assert response.status_code == 200
        body = response.get_json()
        assert "hpi" in body
        assert isinstance(body["hpi"], (int, float))

    def test_invalid_payload_returns_400_with_error_message(self, client, auth_headers):
        payload = {"country": "FRANCE", "year": 2022, "hicp": 105.0, "unemployment_rate": 7.5}
        response = client.post("/predict", json=payload, headers=auth_headers)

        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_non_json_body_returns_400(self, client, auth_headers):
        response = client.post(
            "/predict", data="not json", content_type="text/plain", headers=auth_headers
        )
        assert response.status_code == 400

    def test_missing_field_returns_400(self, client, auth_headers):
        payload = {"country": "FR", "year": 2022, "hicp": 105.0}  # missing unemployment_rate
        response = client.post("/predict", json=payload, headers=auth_headers)
        assert response.status_code == 400
