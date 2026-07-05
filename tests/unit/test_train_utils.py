"""Unit tests for train.py data validation and utility functions.

These tests are fast, hermetic (no network, no MLflow server, no DVC remote)
and check the pure-logic parts of the training pipeline.

Run: pytest tests/unit/test_train_utils.py -v
"""
import pytest

from train import get_dvc_data_hash, get_git_commit, load_dataset

pytestmark = pytest.mark.unit


class TestLoadDataset:
    """load_dataset() must validate schema and minimum size before training."""

    def test_valid_dataset_loads_successfully(self, sample_csv):
        df = load_dataset(str(sample_csv))
        assert len(df) > 0
        assert "hpi" in df.columns

    def test_missing_required_column_raises(self, tmp_path, missing_columns_dataframe):
        csv_path = tmp_path / "bad.csv"
        missing_columns_dataframe.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="missing columns"):
            load_dataset(str(csv_path))

    def test_dataset_below_minimum_size_raises(self, tmp_path, tiny_dataframe):
        csv_path = tmp_path / "tiny.csv"
        tiny_dataframe.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="too small"):
            load_dataset(str(csv_path))

    def test_rows_with_nulls_in_required_columns_are_dropped(self, tmp_path, sample_dataframe):
        df = sample_dataframe.copy()
        df.loc[0, "hpi"] = None
        csv_path = tmp_path / "with_nulls.csv"
        df.to_csv(csv_path, index=False)

        loaded = load_dataset(str(csv_path))
        assert loaded["hpi"].isna().sum() == 0
        assert len(loaded) == len(df) - 1


class TestGetDvcDataHash:
    """Traceability: every training run should record which DVC data version was used."""

    def test_returns_hash_when_dvc_file_exists(self, sample_csv_with_dvc_file):
        result = get_dvc_data_hash(str(sample_csv_with_dvc_file))
        assert result == "b52b097a27380a9510dc5344da77b6"

    def test_returns_unknown_when_dvc_file_missing(self, sample_csv):
        # sample_csv fixture has no .dvc sidecar file
        result = get_dvc_data_hash(str(sample_csv))
        assert result == "unknown"

    def test_returns_unknown_on_malformed_dvc_file(self, tmp_path, sample_csv):
        dvc_path = tmp_path / f"{sample_csv.name}.dvc"
        dvc_path.write_text("not: [valid, dvc, structure}")
        result = get_dvc_data_hash(str(sample_csv))
        assert result == "unknown"


class TestGetGitCommit:
    """Traceability: every run should record the code version (git commit)."""

    def test_returns_non_empty_string(self):
        # In a real git repo this returns a 40-char SHA; outside one it returns "unknown".
        # Either way it must never raise and must return a string.
        result = get_git_commit()
        assert isinstance(result, str)
        assert len(result) > 0
