import pytest

from backend.api import app

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_predict_valid_input_returns_hpi(client, auth_headers):
    payload = {"country": "AT", "year": 2015, "hicp": 0.8, "unemployment_rate": 6.1}
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.get_json()
    assert "hpi" in data
    assert isinstance(data["hpi"], (int, float))


def test_predict_invalid_input_returns_400(client, auth_headers):
    payload = {"country": "Austria", "year": 2015, "hicp": 0.8, "unemployment_rate": 6.1}
    response = client.post("/predict", json=payload, headers=auth_headers)
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_missing_body_returns_400(client, auth_headers):
    response = client.post(
        "/predict", data="not json", content_type="text/plain", headers=auth_headers
    )
    assert response.status_code == 400
