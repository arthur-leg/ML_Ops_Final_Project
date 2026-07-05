import os

import pandas as pd
import pytest

from backend.data_loader import REQUIRED_COLUMNS, download_csv, load_data

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_csv(tmp_path):
    csv_content = (
        "country,year,hpi,hicp,unemployment_rate\n"
        "AT,2015,100.0,0.8,6.1\n"
        "BE,2015,100.0,0.6,8.7\n"
        "BG,2015,100.0,-1.1,10.1\n"
    )
    csv_path = tmp_path / "sample_data.csv"
    csv_path.write_text(csv_content)
    return str(csv_path)


def test_load_data_returns_expected_schema(sample_csv):
    df = load_data(sample_csv)
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 3
    assert pd.api.types.is_numeric_dtype(df["hpi"])


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data("data/does_not_exist.csv")


def test_load_data_missing_columns_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("country,year\nAT,2015\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_data(str(bad_csv))


def test_download_csv_writes_file_from_drive(monkeypatch, tmp_path):
    """Mocks the actual network call to gdown, but exercises the real
    wiring: URL construction, directory creation, and file write path.
    We avoid a real network hit here to keep CI fast and non-flaky --
    the real Drive link should be checked manually or in a nightly job.
    """
    calls = {}

    def fake_download(url, output, quiet):
        calls["url"] = url
        with open(output, "w") as f:
            f.write("country,year,hpi,hicp,unemployment_rate\nAT,2015,100.0,0.8,6.1\n")

    monkeypatch.setattr("backend.data_loader.gdown.download", fake_download)

    output_path = str(tmp_path / "raw_data.csv")
    result_path = download_csv("fake_file_id", output_path)

    assert result_path == output_path
    assert os.path.exists(output_path)
    assert "fake_file_id" in calls["url"]
