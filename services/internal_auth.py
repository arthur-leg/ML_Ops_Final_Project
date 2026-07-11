"""Shared auth for internal microservices (train_service, promote_service).

These are called by CI or by other backend services, never by a browser,
so a static API key checked via a constant-time comparison is enough --
no need to reuse the user-facing Google/JWT flow here.
"""
import hmac
import os
from functools import wraps

from flask import jsonify, request

INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "dev-internal-key-change-me")


def require_internal_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        provided = request.headers.get("X-Internal-Key", "")
        if not hmac.compare_digest(provided, INTERNAL_API_KEY):
            return jsonify({"error": "unauthorized", "message": "Invalid or missing X-Internal-Key header"}), 401
        return f(*args, **kwargs)

    return decorated
