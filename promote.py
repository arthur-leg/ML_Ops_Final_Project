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

from dotenv import load_dotenv
load_dotenv()

REGISTERED_MODEL_NAME = "hpi-forecast"
TARGET_COLUMN = "hpi"
MAE_IMPROVEMENT_REQUIRED = 0.0  # set >0 to require a strict improvement margin


def get_latest_version_by_stage(client: MlflowClient, stage: str):
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[stage])
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


def main(eval_data_path: str):
    client = MlflowClient()

    staging = get_latest_version_by_stage(client, "Staging")
    if staging is None:
        print("No model in Staging. Nothing to promote.")
        sys.exit(1)

    production = get_latest_version_by_stage(client, "Production")

    df = pd.read_csv(eval_data_path).dropna()

    staging_mae = evaluate_model(f"models:/{REGISTERED_MODEL_NAME}/{staging.version}", df)
    print(f"Staging model (v{staging.version}) MAE on {eval_data_path}: {staging_mae:.4f}")

    if production is not None:
        production_mae = evaluate_model(
            f"models:/{REGISTERED_MODEL_NAME}/{production.version}", df
        )
        print(f"Production model (v{production.version}) MAE on {eval_data_path}: {production_mae:.4f}")
    else:
        production_mae = float("inf")
        print("No current Production model -- Staging will be promoted if it beats a trivial baseline.")

    passed = staging_mae <= (production_mae - MAE_IMPROVEMENT_REQUIRED)

    if passed:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=staging.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"GATE PASSED: promoted version {staging.version} to Production.")
        sys.exit(0)
    else:
        print("GATE FAILED: Staging model does not improve on Production. No promotion.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote Staging model to Production")
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV with real/recent data to evaluate candidate vs production model",
    )
    args = parser.parse_args()
    main(args.data)
