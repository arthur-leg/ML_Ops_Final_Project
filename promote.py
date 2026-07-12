"""Promote the latest Staging model to Production if it passes quality gates.

Usage:
    python promote.py --data data/hpi_recent.csv

Logic:
    1. Load the latest model version currently in "Staging" from the MLflow registry.
    2. Load the current "Production" model (if any).
    3. Evaluate both on the given dataset (ideally the most recent real data,
       not seen during training -- e.g. actuals for years after the training cutoff).
    4. If Staging's MAE is better than (or no) Production's MAE:
         -> transition Staging model to "Production"
         -> archive the previous Production version
       Else:
         -> leave Staging as-is, Production unchanged
         -> exit with non-zero status code, so CI blocks the deploy step

This script is meant to be called from the "dev -> staging" and
"staging -> main" CI pipelines as the promotion gate.
"""
import argparse
import sys

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from backend.preprocessing import encode_features

REGISTERED_MODEL_NAME = "hpi-forecast"
TARGET_COLUMN = "hpi"
MAE_IMPROVEMENT_REQUIRED = 0.0  # set >0 to require a strict improvement margin


def get_latest_version_by_stage(client: MlflowClient, stage: str):
    try:
        versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[stage])
    except mlflow.exceptions.MlflowException:
        # Registered model doesn't exist at all yet (e.g. no training run
        # has ever been logged) -- treat the same as "no version in this stage".
        return None
    return versions[0] if versions else None


def evaluate_model(model_uri: str, df: pd.DataFrame) -> float:
    model = mlflow.sklearn.load_model(model_uri)
    X = encode_features(df)
    y = df[TARGET_COLUMN]

    # aligne les colonnes avec celles vues a l'entrainement (si le modele
    # a ete entraine avec un jeu de colonnes different, ex. nouveaux pays)
    expected_cols = model.feature_names_in_
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]

    predictions = model.predict(X)
    return float(mean_absolute_error(y, predictions))


def promote(eval_data_path: str) -> dict:
    """Run the promotion gate and return a result dict. Raises no exceptions
    for a normal 'gate failed' outcome -- that's a valid result, not an error.
    Callable both from the CLI (__main__) and from the promote microservice.
    """
    client = MlflowClient()

    staging = get_latest_version_by_stage(client, "Staging")
    if staging is None:
        return {"promoted": False, "reason": "no_staging_model"}

    production = get_latest_version_by_stage(client, "Production")

    df = pd.read_csv(eval_data_path).dropna()

    staging_mae = evaluate_model(f"models:/{REGISTERED_MODEL_NAME}/{staging.version}", df)

    if production is not None:
        production_mae = evaluate_model(
            f"models:/{REGISTERED_MODEL_NAME}/{production.version}", df
        )
    else:
        production_mae = float("inf")

    passed = staging_mae <= (production_mae - MAE_IMPROVEMENT_REQUIRED)

    result = {
        "staging_version": staging.version,
        "staging_mae": round(staging_mae, 4),
        "production_version": production.version if production else None,
        "production_mae": None if production is None else round(production_mae, 4),
    }

    if passed:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=staging.version,
            stage="Production",
            archive_existing_versions=True,
        )
        result.update({"promoted": True, "reason": "gate_passed"})
    else:
        result.update({"promoted": False, "reason": "gate_failed"})

    return result


def main(eval_data_path: str) -> None:
    outcome = promote(eval_data_path)
    print(outcome)
    sys.exit(0 if outcome["promoted"] else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote Staging model to Production")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV with real/recent data to evaluate candidate vs production model",
    )
    args = parser.parse_args()
    main(args.data)
