"""Loads the current Production model from the MLflow registry.

Import this in backend/api.py instead of loading models/model.joblib directly.
Reloading is cheap enough to do lazily / on a timer if you want the API to
pick up new Production promotions without a redeploy -- kept simple here
(load once at startup) since the project deploys a new container per stage anyway.
"""
import mlflow
import mlflow.sklearn

REGISTERED_MODEL_NAME = "hpi-forecast"


def load_production_model():
    """Load the current Production-stage model. Raises if none exists."""
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
    model = mlflow.sklearn.load_model(model_uri)
    return model


# --- usage dans backend/api.py ---
#
# from backend.model_loader import load_production_model
#
# model = load_production_model()  # charge au demarrage du process Flask
#
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.get_json()
#     df = pd.DataFrame([data])
#     X = encode_features(df)
#     # aligner les colonnes comme dans promote.py si necessaire
#     prediction = model.predict(X)
#     return jsonify({"prediction": float(prediction[0])})
