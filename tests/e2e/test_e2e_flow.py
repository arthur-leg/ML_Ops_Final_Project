"""End-to-end test: hits a fully running instance of the app over HTTP.

Run this against a docker-compose'd local stack or the deployed staging
environment -- NOT the Flask test client. Point it at the right host via
the APP_BASE_URL env var:

    APP_BASE_URL=http://localhost:5000 pytest -m e2e
"""
import os

import pytest
import requests

pytestmark = pytest.mark.e2e

BASE_URL = os.environ.get("APP_BASE_URL", "http://localhost:5000")


def test_full_predict_flow_health_and_prediction():
    health = requests.get(f"{BASE_URL}/health", timeout=5)
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    payload = {"country": "AT", "year": 2020, "hicp": 1.2, "unemployment_rate": 5.4}
    response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
    assert response.status_code == 200

    body = response.json()
    assert "hpi" in body
    assert isinstance(body["hpi"], (int, float))
