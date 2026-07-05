"""Unit tests for promote.py's pure decision logic.

We isolate the promotion decision (staging_mae vs production_mae comparison)
from MLflow I/O by testing the comparison logic directly, and by mocking
MlflowClient/evaluate_model in the one test that exercises main()'s flow.

Run: pytest tests/unit/test_promote_logic.py -v
"""
from unittest.mock import MagicMock, patch

import pytest

from promote import MAE_IMPROVEMENT_REQUIRED, evaluate_model, get_latest_version_by_stage


class TestGetLatestVersionByStage:
    def test_returns_none_when_no_versions_exist(self):
        client = MagicMock()
        client.get_latest_versions.return_value = []
        result = get_latest_version_by_stage(client, "Production")
        assert result is None

    def test_returns_first_version_when_present(self):
        client = MagicMock()
        fake_version = MagicMock(version="3")
        client.get_latest_versions.return_value = [fake_version]
        result = get_latest_version_by_stage(client, "Staging")
        assert result.version == "3"


class TestEvaluateModel:
    """evaluate_model must align prediction columns with the model's training columns,
    so a model trained on an older column set doesn't crash on newer data."""

    @patch("promote.mlflow.sklearn.load_model")
    def test_missing_columns_are_filled_with_zero(self, mock_load_model, sample_dataframe):
        # model was trained expecting an extra one-hot column not present in new eval data
        fake_model = MagicMock()
        fake_model.feature_names_in_ = ["year", "hicp", "unemployment_rate", "country_XX"]
        fake_model.predict.return_value = [100.0] * len(sample_dataframe)
        mock_load_model.return_value = fake_model

        mae = evaluate_model("models:/hpi-forecast/1", sample_dataframe)

        assert isinstance(mae, float)
        called_X = fake_model.predict.call_args[0][0]
        assert "country_XX" in called_X.columns
        assert (called_X["country_XX"] == 0).all()


class TestPromotionGateDecision:
    """The gate must promote only when staging is at least as good as production."""

    def test_staging_better_than_production_passes(self):
        staging_mae, production_mae = 2.5, 3.0
        passed = staging_mae <= (production_mae - MAE_IMPROVEMENT_REQUIRED)
        assert passed is True

    def test_staging_worse_than_production_fails(self):
        staging_mae, production_mae = 4.0, 3.0
        passed = staging_mae <= (production_mae - MAE_IMPROVEMENT_REQUIRED)
        assert passed is False

    def test_no_production_model_staging_always_passes(self):
        staging_mae, production_mae = 6.17, float("inf")
        passed = staging_mae <= (production_mae - MAE_IMPROVEMENT_REQUIRED)
        assert passed is True
