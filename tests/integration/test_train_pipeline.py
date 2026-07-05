"""Integration tests: train() end-to-end against a local, temporary MLflow
tracking server (file-based backend), so tests don't depend on network
access or the real DagsHub instance.

These are slower than unit tests (they actually fit a RandomForest and
write files), but still run in seconds on a small dataset.

Run: pytest tests/integration/test_train_pipeline.py -v
"""
import json

import mlflow
import pytest

import train as train_module

pytestmark = pytest.mark.integration


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    """Point MLflow at a throwaway local SQLite-backed tracking server instead
    of DagsHub, and redirect the model output dir to a temp folder.

    NOTE: the plain filesystem backend (file:./mlruns) no longer supports
    the Model Registry in recent MLflow versions -- SQLite is the simplest
    local backend that still does.
    """
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    model_dir = tmp_path / "models"
    monkeypatch.setattr(train_module, "MODEL_DIR", model_dir)

    yield db_path


class TestTrainEndToEnd:
    def test_train_produces_expected_metrics_keys(self, local_mlflow, sample_csv):
        result = train_module.train(str(sample_csv))

        assert "metrics" in result
        assert set(result["metrics"].keys()) >= {"mae", "r2", "n_train", "n_test"}
        assert isinstance(result["metrics"]["mae"], float)
        assert result["metrics"]["mae"] >= 0

    def test_train_registers_model_in_mlflow_registry(self, local_mlflow, sample_csv):
        train_module.train(str(sample_csv))

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(
            f"name='{train_module.REGISTERED_MODEL_NAME}'"
        )
        assert len(versions) >= 1

    def test_train_writes_local_metrics_json_with_traceability_fields(
        self, local_mlflow, sample_csv
    ):
        train_module.train(str(sample_csv))

        metrics_path = train_module.MODEL_DIR / "metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as f:
            content = json.load(f)

        # Traceability requirement from the project spec: every run must be
        # traceable to a data version and a code (git) version.
        assert "git_commit" in content
        assert "dvc_data_hash" in content
        assert "mlflow_run_id" in content

    def test_train_raises_on_invalid_dataset(self, local_mlflow, tmp_path):
        bad_csv = tmp_path / "empty.csv"
        bad_csv.write_text("country,year,hicp,unemployment_rate,hpi\n")

        with pytest.raises(ValueError):
            train_module.train(str(bad_csv))
