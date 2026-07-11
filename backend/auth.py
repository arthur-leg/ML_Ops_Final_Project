"""Google OAuth login + JWT issuance/verification for the backend."""
import datetime
import os
from functools import wraps

import jwt
from flask import Blueprint, current_app, g, jsonify, request
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from dotenv import load_dotenv

from backend.app_db import upsert_user

load_dotenv()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

auth_bp = Blueprint("auth", __name__)


def _issue_jwt(email: str, name: str) -> str:
    payload = {
        "email": email,
        "name": name,
        "scope": "user",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def generate_external_token(client_name: str, expires_hours: int = 24 * 30) -> str:
    """Issue a JWT for an external (non-Google-authenticated) API client.

    Not exposed as a public endpoint on purpose: an external partner should
    receive this token out-of-band (e.g. handed to them, or generated via
    the CLI script), not be able to self-issue one by hitting a route.
    """
    payload = {
        "client": client_name,
        "scope": "external",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=expires_hours),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


@auth_bp.route("/auth/google", methods=["POST"])
def google_login():
    payload = request.get_json(silent=True) or {}
    google_token = payload.get("token")
    if not google_token:
        return jsonify({"error": "Missing Google token"}), 400

    if not GOOGLE_CLIENT_ID:
        return jsonify({"error": "Server is missing GOOGLE_CLIENT_ID configuration"}), 500

    try:
        idinfo = id_token.verify_oauth2_token(
            google_token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
    except ValueError:
        return jsonify({"error": "Invalid Google token"}), 401

    email = idinfo.get("email")
    name = idinfo.get("name", email)

    if not email:
        return jsonify({"error": "Google token did not contain an email"}), 401

    try:
        user_record = upsert_user(email=email, name=name)
    except Exception:
        current_app.logger.exception("Failed to persist Google-authenticated user")
        return jsonify({"error": "User persistence failed"}), 500

    access_token = _issue_jwt(email, name)

    return jsonify({"access_token": access_token, "email": email, "name": name, "user": user_record}), 200


def _decode_token():
    """Shared token-parsing logic. Returns (payload, None) or (None, (response, status))."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None, (jsonify({"error": "missing_token", "message": "Authorization header (Bearer <token>) required"}), 401)

    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None, (jsonify({"error": "token_expired", "message": "Token expired"}), 401)
    except jwt.InvalidTokenError:
        return None, (jsonify({"error": "invalid_token", "message": "Invalid token"}), 401)

    return payload, None


def require_auth(f):
    """Protects a route behind any valid JWT issued by /auth/google (scope=user)."""

    @wraps(f)
    def decorated(*args, **kwargs):
        payload, error = _decode_token()
        if error:
            return error

        g.current_user = payload
        return f(*args, **kwargs)

    return decorated


def require_scope(scope: str):
    """Protects a route behind a JWT carrying a specific scope claim.
    Use require_scope("external") for the external-facing API, so a
    Google-login user token can't accidentally pass, and vice versa."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            payload, error = _decode_token()
            if error:
                return error

            if payload.get("scope") != scope:
                return jsonify({
                    "error": "forbidden",
                    "message": f"This token does not have '{scope}' scope"
                }), 403

            g.current_user = payload
            return f(*args, **kwargs)

        return decorated

    return decorator