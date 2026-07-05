"""Shared pytest fixtures for unit/integration/e2e tests."""
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe():
    """A small but valid dataset matching the expected training schema.

    NOTE: adjust column names here if backend.preprocessing.REQUIRED_FEATURE_COLUMNS
    differs from ["country", "year", "hicp", "unemployment_rate"] + target "hpi".
    """
    rows = []
    countries = ["FR", "DE", "BE", "ES"]
    for country in countries:
        for year in range(2015, 2021):
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "hicp": 100 + (year - 2015) * 1.5,
                    "unemployment_rate": 8.0 + (year - 2015) * 0.1,
                    "hpi": 100 + (year - 2015) * 5,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_csv(tmp_path, sample_dataframe):
    """Write the sample dataframe to a temp CSV and return its path."""
    csv_path = tmp_path / "training_table.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_csv_with_dvc_file(sample_csv):
    """Same as sample_csv, plus a fake .dvc sidecar file for hash extraction tests."""
    dvc_content = {
        "outs": [
            {
                "md5": "b52b097a27380a9510dc5344da77b6",
                "size": 12345,
                "path": sample_csv.name,
            }
        ]
    }
    dvc_path = Path(f"{sample_csv}.dvc")
    import yaml

    with open(dvc_path, "w") as f:
        yaml.safe_dump(dvc_content, f)
    return sample_csv


@pytest.fixture
def tiny_dataframe():
    """Dataset too small to pass the minimum-rows check (< 20 rows)."""
    return pd.DataFrame(
        {
            "country": ["FR"] * 5,
            "year": list(range(2015, 2020)),
            "hicp": [100.0] * 5,
            "unemployment_rate": [8.0] * 5,
            "hpi": [100.0] * 5,
        }
    )


@pytest.fixture
def missing_columns_dataframe():
    """Dataset missing a required column (unemployment_rate)."""
    return pd.DataFrame(
        {
            "country": ["FR"] * 25,
            "year": list(range(2000, 2025)),
            "hicp": [100.0] * 25,
            "hpi": [100.0] * 25,
        }
    )
