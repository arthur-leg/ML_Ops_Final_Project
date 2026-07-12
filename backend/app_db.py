"""Application database helpers for user accounts.

This module is intentionally separate from the MLflow tracking registry.
It stores application users in a small relational table keyed by email.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

APP_DATABASE_URL = os.environ.get("APP_DATABASE_URL") or os.environ.get("DATABASE_URL")
SCHEMA_PATH = Path(__file__).resolve().parent / "sql" / "001_create_users.sql"


def _normalize_sqlite_path(database_url: str) -> str:
    if database_url == "sqlite:///:memory:":
        return ":memory:"
    if database_url.startswith("sqlite:///"):
        return database_url.replace("sqlite:///", "", 1)
    return database_url


def _is_sqlite(database_url: str) -> bool:
    return database_url.startswith("sqlite:")


def _connect():
    if not APP_DATABASE_URL:
        raise RuntimeError(
            "Application database is not configured. Set APP_DATABASE_URL to a Postgres or SQLite URL."
        )

    if _is_sqlite(APP_DATABASE_URL):
        sqlite_path = _normalize_sqlite_path(APP_DATABASE_URL)
        sqlite_parent = Path(sqlite_path).expanduser().resolve().parent
        if sqlite_path != ":memory:":
            sqlite_parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(sqlite_path)
        connection.row_factory = sqlite3.Row
        return connection

    import psycopg2
    from psycopg2.extras import RealDictCursor

    return psycopg2.connect(APP_DATABASE_URL, cursor_factory=RealDictCursor)


def _placeholder(database_url: str) -> str:
    return "?" if _is_sqlite(database_url) else "%s"


def _load_schema_sql() -> str:
    with open(SCHEMA_PATH, encoding="utf-8") as schema_file:
        return schema_file.read()


def ensure_schema() -> None:
    """Apply the explicit schema migration for the users table if needed."""
    connection = _connect()
    try:
        with closing(connection.cursor()) as cursor:
            cursor.execute(_load_schema_sql())
        connection.commit()
    finally:
        connection.close()


def upsert_user(email: str, name: str) -> dict[str, Any]:
    """Insert or refresh the user record for a Google-authenticated account."""
    if not email:
        raise ValueError("email is required")

    ensure_schema()
    connection = _connect()
    try:
        database_url = APP_DATABASE_URL or ""
        email_placeholder = _placeholder(database_url)
        name_placeholder = _placeholder(database_url)
        if _is_sqlite(database_url):
            with closing(connection.cursor()) as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (email, name)
                    VALUES (?, ?)
                    ON CONFLICT(email) DO UPDATE SET
                        name = excluded.name
                    """,
                    (email, name),
                )
                cursor.execute(
                    "SELECT email, name, created_at FROM users WHERE email = ?",
                    (email,),
                )
                row = cursor.fetchone()
        else:
            with closing(connection.cursor()) as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO users (email, name)
                    VALUES ({email_placeholder}, {name_placeholder})
                    ON CONFLICT (email) DO UPDATE SET
                        name = EXCLUDED.name
                    """,
                    (email, name),
                )
                cursor.execute(
                    f"SELECT email, name, created_at FROM users WHERE email = {email_placeholder}",
                    (email,),
                )
                row = cursor.fetchone()

        connection.commit()
        if row is None:
            raise RuntimeError("Failed to persist user record")
        return dict(row)
    finally:
        connection.close()


def get_user(email: str) -> dict[str, Any] | None:
    """Return a single user record by email, or None if it does not exist."""
    if not APP_DATABASE_URL:
        return None

    ensure_schema()
    connection = _connect()
    try:
        with closing(connection.cursor()) as cursor:
            placeholder = _placeholder(APP_DATABASE_URL)
            cursor.execute(
                f"SELECT email, name, created_at FROM users WHERE email = {placeholder}",
                (email,),
            )
            row = cursor.fetchone()
        return dict(row) if row is not None else None
    finally:
        connection.close()
