"""Train microservice -- exposes the model training step as its own API.

Usage:
    python -m services.train_service.app
    (runs on port 5001 by default)

Endpoint:
    POST /train
    Headers: X-Internal-Key: <INTERNAL_API_KEY>
    Body (JSON, optional): {"data_path": "data/hpi.csv", "n_estimators": 100}
"""
import os

from flask import Flask, jsonify, request

from services.internal_auth import require_internal_key
from train import train

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "train_service"}), 200


@app.route("/train", methods=["POST"])
@require_internal_key
def run_training():
    payload = request.get_json(silent=True) or {}
    data_path = payload.get("data_path", "data/hpi.csv")
    n_estimators = payload.get("n_estimators", 100)

    try:
        result = train(data_path, n_estimators=n_estimators)
    except (ValueError, FileNotFoundError) as e:
        return jsonify({"error": "training_failed", "message": str(e)}), 400
    except Exception:
        app.logger.exception("Unexpected error during training")
        return jsonify({"error": "internal_error", "message": "Training failed unexpectedly"}), 500

    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("TRAIN_SERVICE_PORT", 5001)))
