"""End-to-end test: hits a fully running instance of the app over HTTP.

Two modes:

1. APP_BASE_URL set -> tests that remote/already-running server
   (docker-compose stack, deployed staging, ...):

       APP_BASE_URL=http://localhost:5000 pytest -m e2e

2. APP_BASE_URL not set (default, used in CI) -> the fixture boots a
   real local Flask server in a subprocess, so the test is fully
   self-contained and can run automatically in the pipeline.
"""
import os
import signal
import subprocess
import sys
import time

import pytest
import requests

pytestmark = pytest.mark.e2e

REMOTE_BASE_URL = os.environ.get("APP_BASE_URL")
LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 5001


def wait_for_server(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(url, timeout=1)
            return
        except requests.exceptions.ConnectionError:
            time.sleep(0.3)
    raise RuntimeError(f"Server at {url} did not start within {timeout}s")


@pytest.fixture(scope="module")
def base_url():
    if REMOTE_BASE_URL:
        # Mode 1: test an already-running server
        yield REMOTE_BASE_URL
        return

    # Mode 2: boot a real local server for the duration of the tests
    env = {**os.environ, "FLASK_APP": "backend.api"}
    process = subprocess.Popen(
        [sys.executable, "-m", "flask", "run", "--host", LOCAL_HOST, "--port", str(LOCAL_PORT)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    url = f"http://{LOCAL_HOST}:{LOCAL_PORT}"
    try:
        wait_for_server(f"{url}/health")
        yield url
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=10)


def test_full_predict_flow_health_and_prediction(base_url, auth_headers):
    health = requests.get(f"{base_url}/health", timeout=5)
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    payload = {"country": "AT", "year": 2020, "hicp": 1.2, "unemployment_rate": 5.4}
    response = requests.post(f"{base_url}/predict", json=payload, headers=auth_headers, timeout=5)
    assert response.status_code == 200

    body = response.json()
    assert "hpi" in body
    assert isinstance(body["hpi"], (int, float))


def test_invalid_input_rejected_by_live_server(base_url, auth_headers):
    response = requests.post(
        f"{base_url}/predict", json={"country": "Austria"}, headers=auth_headers, timeout=5
    )
    assert response.status_code == 400
    assert "error" in response.json()
