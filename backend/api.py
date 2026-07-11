"""Flask API serving HPI predictions.

Loads the current Production model from the MLflow Model Registry
(hosted on DagsHub). Falls back to a placeholder prediction if no
Production model exists yet (e.g. in CI before the first promotion),
so the endpoint contract keeps working either way.
"""
import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, g, jsonify, request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

from backend.preprocessing import encode_features, validate_row
from backend.auth import auth_bp, require_auth
from dotenv import load_dotenv

load_dotenv()

REGISTERED_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "hpi-forecast")
MODEL_STAGE = os.environ.get("MLFLOW_MODEL_STAGE", "Production")
PROMETHEUS_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")

app = Flask(__name__)
app.register_blueprint(auth_bp)

PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total number of prediction requests served by the backend.",
)
PREDICTION_FAILED_REQUESTS_TOTAL = Counter(
    "prediction_failed_requests_total",
    "Total number of failed prediction requests served by the backend.",
)
PREDICTION_REQUEST_LATENCY_SECONDS = Histogram(
    "prediction_request_latency_seconds",
    "Time spent serving prediction requests.",
)
BACKEND_UPTIME_SECONDS = Gauge(
    "backend_uptime_seconds",
    "Backend uptime in seconds since the application process started.",
)
BACKEND_START_TIME_SECONDS = Gauge(
    "backend_start_time_seconds",
    "Unix timestamp when the backend process started.",
)
BACKEND_HEALTH_STATUS = Gauge(
    "backend_health_status",
    "Backend health status, where 1 means the application is serving requests.",
)
BACKEND_MODEL_LOADED = Gauge(
    "backend_model_loaded",
    "Whether the Production model artifact is currently loaded.",
)

BACKEND_START_TIME = time.time()
BACKEND_HEALTH_STATUS.set(1)
BACKEND_MODEL_LOADED.set(0)
BACKEND_UPTIME_SECONDS.set(0)
BACKEND_START_TIME_SECONDS.set(BACKEND_START_TIME)


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
BACKEND_MODEL_LOADED.set(1 if MODEL_ARTIFACT is not None else 0)


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


def _metrics_registry():
    if PROMETHEUS_MULTIPROC_DIR:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return None


@app.before_request
def start_prediction_timer():
    if request.path == "/predict" and request.method == "POST":
        g.prediction_start_time = time.perf_counter()


@app.after_request
def record_prediction_metrics(response):
    if request.path == "/predict" and request.method == "POST":
        PREDICTION_REQUESTS_TOTAL.inc()
        start_time = getattr(g, "prediction_start_time", None)
        if start_time is not None:
            PREDICTION_REQUEST_LATENCY_SECONDS.observe(time.perf_counter() - start_time)
        if 400 <= response.status_code < 500:
            PREDICTION_FAILED_REQUESTS_TOTAL.inc()
    return response


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = os.environ.get(
        "CORS_ALLOW_ORIGIN", "*"
    )
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/metrics", methods=["GET"])
def metrics():
    BACKEND_UPTIME_SECONDS.set(time.time() - BACKEND_START_TIME)
    registry = _metrics_registry()
    if registry is None:
        payload = generate_latest()
    else:
        payload = generate_latest(registry)
    return payload, 200, {"Content-Type": CONTENT_TYPE_LATEST}


@app.route("/health", methods=["GET"])
def health():
    BACKEND_UPTIME_SECONDS.set(time.time() - BACKEND_START_TIME)
    return jsonify(
        {
            "status": "ok",
            "model_loaded": MODEL_ARTIFACT is not None,
            "model_name": REGISTERED_MODEL_NAME,
            "model_stage": MODEL_STAGE,
        }
    ), 200


@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        validate_row(payload)
        prediction = predict_row(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception:
        app.logger.exception("Unexpected error while serving prediction")
        PREDICTION_FAILED_REQUESTS_TOTAL.inc()
        return jsonify({"error": "Internal server error"}), 500

    return jsonify({"hpi": prediction}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
