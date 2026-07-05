"""Flask API serving HPI predictions.

Loads the current Production model from the MLflow Model Registry
(hosted on DagsHub). Falls back to a placeholder prediction if no
Production model exists yet (e.g. in CI before the first promotion),
so the endpoint contract keeps working either way.
"""
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, jsonify, request

from backend.preprocessing import encode_features, validate_row

REGISTERED_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "hpi-forecast")
MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "Production")

app = Flask(__name__)


def load_production_model_artifact():
    """Load the current Production-stage model + its training columns from MLflow.

    Returns a dict {"model": <sklearn model>, "columns": [...]}, matching
    the shape predict_row() expects -- or None if unavailable (e.g. no
    MLFLOW_TRACKING_URI configured, or no model has been promoted yet).
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None

    mlflow.set_tracking_uri(tracking_uri)

    try:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)

        # columns.json was logged as an artifact alongside the model run;
        # fetch it via the registry -> run_id -> artifact path.
        client = mlflow.tracking.MlflowClient()
        version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[MODEL_STAGE])[0]
        run_id = version.run_id
        local_path = client.download_artifacts(run_id, "columns.json")

        import json

        with open(local_path) as f:
            columns = json.load(f)

        return {"model": model, "columns": columns}
    except Exception as e:
        app.logger.warning(f"Could not load Production model from MLflow: {e}")
        return None


MODEL_ARTIFACT = load_production_model_artifact()


def predict_placeholder(row):
    """Fallback when no Production model is available."""
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


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = os.environ.get(
        "CORS_ALLOW_ORIGIN", "*"
    )
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL_ARTIFACT is not None,
            "model_name": REGISTERED_MODEL_NAME,
            "model_stage": MODEL_STAGE,
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
