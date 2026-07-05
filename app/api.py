"""Flask API serving HPI predictions.

Loads the trained model from MODEL_PATH (default: models/model.joblib).
If no model file is present (e.g. in CI before training), falls back to
the placeholder so the endpoint contract keeps working. The next step is
to load the model from the MLflow Model Registry instead of a local file.
"""
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from app.preprocessing import encode_features, validate_row

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

app = Flask(__name__)


def load_model_artifact(path: str):
    """Load the trained model artifact, or None if unavailable."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


MODEL_ARTIFACT = load_model_artifact(MODEL_PATH)


def predict_placeholder(row):
    """Fallback when no trained model is available."""
    return 100.0


def predict_row(row: dict) -> float:
    """Predict hpi for a single validated input row."""
    if MODEL_ARTIFACT is None:
        return predict_placeholder(row)

    df = pd.DataFrame([row])
    encoded = encode_features(df)
    # Align one-hot columns with what the model saw during training
    encoded = encoded.reindex(columns=MODEL_ARTIFACT["columns"], fill_value=0)
    return float(MODEL_ARTIFACT["model"].predict(encoded)[0])


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL_ARTIFACT is not None,
        }
    ), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        validate_row(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prediction = predict_row(payload)
    return jsonify({"hpi": prediction}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
