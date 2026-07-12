from __future__ import annotations

import importlib


def test_upsert_user_creates_and_updates_sqlite_user(monkeypatch, tmp_path):
    database_path = tmp_path / "app.db"
    monkeypatch.setenv("APP_DATABASE_URL", f"sqlite:///{database_path}")

    import backend.app_db as app_db

    importlib.reload(app_db)

    created = app_db.upsert_user("alice@example.com", "Alice")
    assert created["email"] == "alice@example.com"
    assert created["name"] == "Alice"
    assert created["created_at"]

    updated = app_db.upsert_user("alice@example.com", "Alice Cooper")
    assert updated["email"] == "alice@example.com"
    assert updated["name"] == "Alice Cooper"
    assert updated["created_at"] == created["created_at"]

    fetched = app_db.get_user("alice@example.com")
    assert fetched is not None
    assert fetched["name"] == "Alice Cooper"


def test_get_user_returns_none_without_database(monkeypatch):
    monkeypatch.delenv("APP_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    import backend.app_db as app_db

    importlib.reload(app_db)

    assert app_db.get_user("missing@example.com") is None