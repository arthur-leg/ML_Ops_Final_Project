"""Promote microservice -- exposes the promotion gate as its own API.

Usage:
    python -m services.promote_service.app
    (runs on port 5002 by default)

Endpoint:
    POST /promote
    Headers: X-Internal-Key: <INTERNAL_API_KEY>
    Body (JSON): {"data_path": "data/hpi_recent.csv"}
"""
import os

from flask import Flask, jsonify, request

from services.internal_auth import require_internal_key
from promote import promote

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "promote_service"}), 200


@app.route("/promote", methods=["POST"])
@require_internal_key
def run_promotion():
    payload = request.get_json(silent=True) or {}
    data_path = payload.get("data_path")
    if not data_path:
        return jsonify({"error": "bad_request", "message": "data_path is required"}), 400

    try:
        result = promote(data_path)
    except FileNotFoundError:
        return jsonify({"error": "not_found", "message": f"No such file: {data_path}"}), 400
    except Exception:
        app.logger.exception("Unexpected error during promotion")
        return jsonify({"error": "internal_error", "message": "Promotion failed unexpectedly"}), 500

    status_code = 200 if result["promoted"] else 409  # 409: gate ran fine, just didn't pass
    return jsonify(result), status_code


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PROMOTE_SERVICE_PORT", 5002)))
