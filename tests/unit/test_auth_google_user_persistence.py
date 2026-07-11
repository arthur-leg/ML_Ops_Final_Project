from __future__ import annotations

import importlib
import sys
import types

from flask import Flask


def test_google_login_persists_user_record(monkeypatch):
    jwt_module = types.ModuleType("jwt")

    class InvalidTokenError(Exception):
        pass

    class ExpiredSignatureError(Exception):
        pass

    jwt_module.InvalidTokenError = InvalidTokenError
    jwt_module.ExpiredSignatureError = ExpiredSignatureError
    jwt_module.encode = lambda payload, secret, algorithm=None: "encoded-token"
    jwt_module.decode = lambda token, secret, algorithms=None: {"scope": "user"}

    dotenv_module = types.ModuleType("dotenv")
    dotenv_module.load_dotenv = lambda: None

    google_module = types.ModuleType("google")
    google_auth_module = types.ModuleType("google.auth")
    google_transport_module = types.ModuleType("google.auth.transport")
    google_requests_module = types.ModuleType("google.auth.transport.requests")
    google_requests_module.Request = object
    google_transport_module.requests = google_requests_module
    google_auth_module.transport = google_transport_module

    google_oauth2_module = types.ModuleType("google.oauth2")
    google_id_token_module = types.ModuleType("google.oauth2.id_token")
    google_id_token_module.verify_oauth2_token = lambda token, request, client_id: {}
    google_oauth2_module.id_token = google_id_token_module

    monkeypatch.setitem(sys.modules, "jwt", jwt_module)
    monkeypatch.setitem(sys.modules, "dotenv", dotenv_module)
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.auth", google_auth_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", google_transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", google_requests_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", google_oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.id_token", google_id_token_module)
    sys.modules.pop("backend.auth", None)

    auth_module = importlib.import_module("backend.auth")

    monkeypatch.setattr(auth_module, "GOOGLE_CLIENT_ID", "test-client-id")
    monkeypatch.setattr(
        auth_module.id_token,
        "verify_oauth2_token",
        lambda token, request, client_id: {
            "email": "alice@example.com",
            "name": "Alice",
        },
    )
    upsert_calls = []

    def fake_upsert_user(email, name):
        upsert_calls.append((email, name))
        return {"email": email, "name": name, "created_at": "2026-07-11T00:00:00+00:00"}

    monkeypatch.setattr(auth_module, "upsert_user", fake_upsert_user)

    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(auth_module.auth_bp)

    with app.test_client() as client:
        response = client.post("/auth/google", json={"token": "google-token"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["email"] == "alice@example.com"
    assert body["user"]["name"] == "Alice"
    assert upsert_calls == [("alice@example.com", "Alice")]