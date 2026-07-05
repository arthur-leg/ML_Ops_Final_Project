"""Train the HPI prediction model.

Usage:
    python train.py --data data/hpi.csv

Trains a scikit-learn regression model to predict hpi (house price
index) from country, year, hicp and unemployment_rate, using the same
encoding as the API (backend.preprocessing.encode_features).

Outputs:
    models/model.joblib   -- dict {"model", "columns", "metrics", "trained_at",
                             "git_commit"} so the API can align one-hot
                             columns at prediction time.
    models/metrics.json   -- metrics + traceability info (git commit),
                             groundwork for MLflow/DVC integration.
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from backend.preprocessing import REQUIRED_FEATURE_COLUMNS, encode_features

TARGET_COLUMN = "hpi"
MODEL_DIR = Path("models")


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


def train(csv_path: str, test_size: float = 0.2, random_state: int = 42) -> dict:
    df = load_dataset(csv_path)

    X = encode_features(df)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    artifact = {
        "model": model,
        "columns": list(X.columns),
        "metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "data_path": csv_path,
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(artifact, MODEL_DIR / "model.joblib")
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump({k: v for k, v in artifact.items() if k != "model"}, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the HPI model")
    parser.add_argument("--data", default="data/hpi.csv", help="Path to the training CSV")
    args = parser.parse_args()

    results = train(args.data)
    print(f"Model trained. MAE={results['mae']}  R2={results['r2']}")
    print(f"Saved to {MODEL_DIR / 'model.joblib'}")
