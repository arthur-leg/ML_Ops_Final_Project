"""Flask API serving HPI predictions.

NOTE: predict_placeholder() is a stand-in until the real model is
trained and loaded from the MLflow Model Registry. The endpoint
contract (request/response shape) is final -- swap the internals of
predict_placeholder (or replace it with a real model.predict call)
without touching the route or the tests that check status codes and
response shape.
"""
from flask import Flask, jsonify, request

from app.preprocessing import validate_row

app = Flask(__name__)


def predict_placeholder(row):
    """Stand-in for the real model. Replace with MLflow model.predict()."""
    return 100.0


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        validate_row(payload)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    prediction = predict_placeholder(payload)
    return jsonify({"hpi": prediction}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
