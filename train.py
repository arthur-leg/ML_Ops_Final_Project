"""Train the HPI prediction model, tracked and registered via MLflow.

Usage:
    python train.py --data data/hpi.csv

Trains a scikit-learn regression model to predict hpi (house price
index) from country, year, hicp and unemployment_rate, using the same
encoding as the API (backend.preprocessing.encode_features).

Each run is logged to MLflow with:
    - params: data path, DVC data version (git hash of the .dvc file), model hyperparams
    - metrics: mae, r2
    - the model itself, registered under "hpi-forecast" in the MLflow Model Registry
      (new versions start in "Staging" -- promotion to "Production" is a
      separate, explicit step done by promote.py after quality gates pass)

Outputs:
    models/metrics.json   -- metrics + traceability info, kept for quick local
                             inspection (README, debugging) alongside MLflow.
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from backend.preprocessing import REQUIRED_FEATURE_COLUMNS, encode_features

from dotenv import load_dotenv
load_dotenv()

TARGET_COLUMN = "hpi"
MODEL_DIR = Path("models")
REGISTERED_MODEL_NAME = "hpi-forecast"
EXPERIMENT_NAME = "hpi-forecast-experiments"


def get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def get_dvc_data_hash(csv_path: str) -> str:
    """Return the DVC md5 hash for the tracked data file, or 'unknown'.

    Reads the corresponding .dvc file (e.g. data/hpi.csv.dvc) which DVC
    generates alongside every tracked file. This is what makes a training
    run traceable to an exact DVC data version, independent of Git history.
    """
    dvc_file = Path(f"{csv_path}.dvc")
    if not dvc_file.exists():
        return "unknown"
    try:
        import yaml

        with open(dvc_file) as f:
            content = yaml.safe_load(f)
        return content["outs"][0]["md5"]
    except Exception:
        return "unknown"


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and sanity-check the training CSV."""
    df = pd.read_csv(csv_path)
    expected = set(REQUIRED_FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df = df.dropna(subset=list(expected))
    if len(df) < 20:
        raise ValueError(f"Dataset too small after cleaning: {len(df)} rows")
    return df


def train(
    csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
) -> dict:
    df = load_dataset(csv_path)

    X = encode_features(df)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    git_commit = get_git_commit()
    dvc_hash = get_dvc_data_hash(csv_path)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        mlflow.log_param("data_path", csv_path)
        mlflow.log_param("dvc_data_hash", dvc_hash)
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("n_features", X.shape[1])

        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        metrics = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
            "r2": round(float(r2_score(y_test, predictions)), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        mlflow.log_metric("mae", metrics["mae"])
        mlflow.log_metric("r2", metrics["r2"])
        mlflow.log_metric("n_train", metrics["n_train"])
        mlflow.log_metric("n_test", metrics["n_test"])

        # --- log + enregistre le modele dans le Model Registry ---
        # Nouvelle version enregistree, elle demarre automatiquement en stage "None"/"Staging".
        # La promotion vers "Production" est geree separement par promote.py,
        # apres verification des quality gates -- jamais automatique ici.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(2),
        )

        # on stocke les colonnes d'encodage comme artifact JSON, car l'API
        # doit reconstruire le meme encodage one-hot a la prediction
        columns_path = MODEL_DIR / "columns.json"
        MODEL_DIR.mkdir(exist_ok=True)
        with open(columns_path, "w") as f:
            json.dump(list(X.columns), f)
        mlflow.log_artifact(str(columns_path))

        run_id = run.info.run_id

    # transition explicite vers "Staging" -- par defaut un modele nouvellement
    # enregistre n'a pas de stage assigne, on le met en Staging volontairement
    # pour respecter le flux : Staging (candidat) -> Production (promu par promote.py)
    client = mlflow.tracking.MlflowClient()
    latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])
    if latest_version:
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=latest_version[0].version,
            stage="Staging",
        )

    # --- copie locale des metriques pour inspection rapide / README ---
    summary = {
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "dvc_data_hash": dvc_hash,
        "data_path": csv_path,
        "mlflow_run_id": run_id,
        "registered_model_name": REGISTERED_MODEL_NAME,
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the HPI model")
    parser.add_argument("--data", default="data/hpi.csv", help="Path to the training CSV")
    parser.add_argument("--n-estimators", type=int, default=100)
    args = parser.parse_args()

    results = train(args.data, n_estimators=args.n_estimators)
    print(f"Model trained. MAE={results['metrics']['mae']}  R2={results['metrics']['r2']}")
    print(f"MLflow run: {results['mlflow_run_id']}")
    print(f"Registered as: {results['registered_model_name']} (stage: Staging by default)")
