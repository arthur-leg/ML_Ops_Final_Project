"""End-to-end test: hits a *running* backend (e.g. the staging deployment)
and verifies the /predict endpoint returns a valid prediction using the
current Production model.

ASSUMPTIONS (adjust if your actual API differs):
  - The Flask app exposes POST /predict, accepting JSON:
        {"country": "FR", "year": 2022, "hicp": 105.0, "unemployment_rate": 8.2}
    and returning JSON:
        {"prediction": <float>}
  - backend/api.py loads the model via backend.model_loader.load_production_model()
    (i.e. from MLflow's "Production" stage), NOT from a local .joblib file.
  - The API base URL is provided via the E2E_BASE_URL env var, defaulting to
    http://localhost:5000 for local runs. In CI, this should point at the
    actual deployed staging URL.

Run locally (with the Flask app already running):
    python -m flask --app backend.api run --host 0.0.0.0 --port 5000
    E2E_BASE_URL=http://localhost:5000 pytest tests/e2e/test_predict_e2e.py -v

This test is skipped automatically if the API isn't reachable, so it doesn't
break local `pytest` runs for people not currently running the server --
but it MUST run (and must pass) in the CI job for the staging deployment.
"""
import os

import pytest
import requests

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:5000")


def _api_is_reachable() -> bool:
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
        return True
    except requests.exceptions.RequestException:
        try:
            # fall back to hitting /predict's OPTIONS or root, in case
            # there is no dedicated /health endpoint yet
            requests.get(BASE_URL, timeout=2)
            return True
        except requests.exceptions.RequestException:
            return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _api_is_reachable(),
        reason=f"API not reachable at {BASE_URL} -- start the Flask app or set E2E_BASE_URL",
    ),
]


class TestPredictEndpointE2E:
    def test_predict_returns_valid_response_shape(self):
        payload = {
            "country": "FR",
            "year": 2022,
            "hicp": 105.0,
            "unemployment_rate": 8.2,
        }
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)

        assert response.status_code == 200
        body = response.json()
        assert "hpi" in body
        assert isinstance(body["hpi"], (int, float))

    def test_predict_rejects_missing_required_fields(self):
        incomplete_payload = {"country": "FR"}  # missing year, hicp, unemployment_rate
        response = requests.post(f"{BASE_URL}/predict", json=incomplete_payload, timeout=10)

        # api.py returns 400 with an {"error": ...} body via validate_row()
        assert response.status_code == 400
        assert "error" in response.json()

    def test_predict_is_consistent_for_same_input(self):
        """Same input should give the same prediction (model is deterministic
        at inference time -- guards against e.g. accidentally reloading a
        different/random model per request)."""
        payload = {
            "country": "DE",
            "year": 2021,
            "hicp": 103.0,
            "unemployment_rate": 7.5,
        }
        r1 = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)
        r2 = requests.post(f"{BASE_URL}/predict", json=payload, timeout=10)

        assert r1.json()["hpi"] == pytest.approx(r2.json()["hpi"])

    def test_health_endpoint_reports_model_status(self):
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        body = response.json()
        assert "status" in body
        assert "model_loaded" in body
