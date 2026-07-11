"""Google OAuth login + JWT issuance/verification for the backend."""
import datetime
import os
from functools import wraps

import jwt
from flask import Blueprint, g, jsonify, request
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

auth_bp = Blueprint("auth", __name__)


def _issue_jwt(email: str, name: str) -> str:
    payload = {
        "email": email,
        "name": name,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=JWT_EXPIRY_HOURS),
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

    access_token = _issue_jwt(email, name)

    return jsonify({"access_token": access_token, "email": email, "name": name}), 200


def require_auth(f):
    """Decorator: protects a Flask route behind a valid JWT (issued by /auth/google
    or requested via the external API with a token)."""

    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ", 1)[1].strip()
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        g.current_user = payload
        return f(*args, **kwargs)

    return decorated