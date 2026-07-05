"""Integration test: the full promotion flow (train -> Staging -> gate -> Production)
against a local, temporary MLflow tracking server.

This exercises the real interaction between train.py's registration and
promote.py's comparison/transition logic, without touching DagsHub.

Run: pytest tests/integration/test_promote_pipeline.py -v
"""
import mlflow
import pytest
from mlflow.tracking import MlflowClient

import promote as promote_module
import train as train_module

pytestmark = pytest.mark.integration


@pytest.fixture
def local_mlflow(tmp_path, monkeypatch):
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    model_dir = tmp_path / "models"
    monkeypatch.setattr(train_module, "MODEL_DIR", model_dir)

    yield db_path


class TestPromotionFlowEndToEnd:
    def test_first_model_is_promoted_when_no_production_exists(
        self, local_mlflow, sample_csv, capsys
    ):
        train_module.train(str(sample_csv))

        with pytest.raises(SystemExit) as exc_info:
            promote_module.main(str(sample_csv))

        assert exc_info.value.code == 0

        client = MlflowClient()
        prod_versions = client.get_latest_versions(
            promote_module.REGISTERED_MODEL_NAME, stages=["Production"]
        )
        assert len(prod_versions) == 1

    def test_no_staging_model_exits_nonzero(self, local_mlflow, sample_csv):
        # No train() call -> registry is empty, nothing in Staging.
        # Need the registered model to exist but have no Staging version;
        # simplest is to assert on a brand-new, empty registry.
        with pytest.raises(SystemExit) as exc_info:
            promote_module.main(str(sample_csv))

        assert exc_info.value.code == 1

    def test_worse_candidate_does_not_replace_better_production(
        self, local_mlflow, sample_csv, tmp_path, sample_dataframe
    ):
        # 1. Train and promote a first (good) model to Production.
        train_module.train(str(sample_csv))
        with pytest.raises(SystemExit):
            promote_module.main(str(sample_csv))

        client = MlflowClient()
        production_before = client.get_latest_versions(
            promote_module.REGISTERED_MODEL_NAME, stages=["Production"]
        )[0].version

        # 2. Train a second model on a degenerate/tiny-signal dataset so it
        #    performs worse, simulating a bad candidate.
        degenerate_df = sample_dataframe.copy()
        degenerate_df["hpi"] = 100.0  # no signal at all -> should generalize worse
        degenerate_csv = tmp_path / "degenerate.csv"
        degenerate_df.to_csv(degenerate_csv, index=False)
        train_module.train(str(degenerate_csv))

        # 3. Evaluate the new Staging candidate against the ORIGINAL data
        #    (where the real model should still win).
        with pytest.raises(SystemExit) as exc_info:
            promote_module.main(str(sample_csv))

        production_after = client.get_latest_versions(
            promote_module.REGISTERED_MODEL_NAME, stages=["Production"]
        )[0].version

        assert exc_info.value.code == 1  # gate failed
        assert production_after == production_before  # production unchanged
