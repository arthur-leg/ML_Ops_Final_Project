"""Unit tests for backend/preprocessing.py.

Covers validate_row() (input validation before prediction/training) and
encode_features() (one-hot encoding used consistently by both train.py
and api.py).

Run: pytest tests/unit/test_preprocessing.py -v
"""
import pandas as pd
import pytest

from backend.preprocessing import encode_features, validate_row

pytestmark = pytest.mark.unit


class TestValidateRow:
    def test_valid_row_passes(self):
        row = {"country": "FR", "year": 2022, "hicp": 105.0, "unemployment_rate": 7.5}
        validate_row(row)  # should not raise

    def test_missing_field_raises(self):
        row = {"country": "FR", "year": 2022, "hicp": 105.0}  # missing unemployment_rate
        with pytest.raises(ValueError, match="Missing required field"):
            validate_row(row)

    def test_null_field_raises(self):
        row = {"country": "FR", "year": 2022, "hicp": None, "unemployment_rate": 7.5}
        with pytest.raises(ValueError, match="Missing required field"):
            validate_row(row)

    def test_invalid_country_code_raises(self):
        row = {"country": "FRA", "year": 2022, "hicp": 105.0, "unemployment_rate": 7.5}
        with pytest.raises(ValueError, match="2-letter ISO code"):
            validate_row(row)

    def test_non_string_country_raises(self):
        row = {"country": 42, "year": 2022, "hicp": 105.0, "unemployment_rate": 7.5}
        with pytest.raises(ValueError, match="2-letter ISO code"):
            validate_row(row)

    def test_year_out_of_range_raises(self):
        row = {"country": "FR", "year": 1800, "hicp": 105.0, "unemployment_rate": 7.5}
        with pytest.raises(ValueError, match="year must be between"):
            validate_row(row)

    def test_non_numeric_year_raises(self):
        row = {"country": "FR", "year": "not-a-year", "hicp": 105.0, "unemployment_rate": 7.5}
        with pytest.raises(ValueError, match="year must be an integer"):
            validate_row(row)

    def test_non_numeric_hicp_raises(self):
        row = {"country": "FR", "year": 2022, "hicp": "abc", "unemployment_rate": 7.5}
        with pytest.raises(ValueError, match="hicp must be numeric"):
            validate_row(row)

    def test_implausible_unemployment_rate_raises(self):
        row = {"country": "FR", "year": 2022, "hicp": 105.0, "unemployment_rate": 250}
        with pytest.raises(ValueError, match="plausible percentage"):
            validate_row(row)

    def test_boundary_years_are_accepted(self):
        for year in (2000, 2035):
            row = {"country": "FR", "year": year, "hicp": 105.0, "unemployment_rate": 7.5}
            validate_row(row)  # should not raise


class TestEncodeFeatures:
    def test_one_hot_encodes_country(self):
        df = pd.DataFrame(
            {
                "country": ["FR", "DE"],
                "year": [2020, 2021],
                "hicp": [100.0, 102.0],
                "unemployment_rate": [8.0, 7.5],
            }
        )
        encoded = encode_features(df)

        assert "country_FR" in encoded.columns
        assert "country_DE" in encoded.columns
        assert encoded.loc[0, "country_FR"] == 1
        assert encoded.loc[0, "country_DE"] == 0

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"country": ["FR"], "year": [2020], "hicp": [100.0]})
        # unemployment_rate missing
        with pytest.raises(ValueError, match="missing columns"):
            encode_features(df)

    def test_numeric_columns_preserved_unchanged(self):
        df = pd.DataFrame(
            {
                "country": ["FR"],
                "year": [2020],
                "hicp": [101.5],
                "unemployment_rate": [8.3],
            }
        )
        encoded = encode_features(df)
        assert encoded.loc[0, "year"] == 2020
        assert encoded.loc[0, "hicp"] == 101.5
        assert encoded.loc[0, "unemployment_rate"] == 8.3

    def test_extra_columns_in_input_are_ignored(self):
        df = pd.DataFrame(
            {
                "country": ["FR"],
                "year": [2020],
                "hicp": [100.0],
                "unemployment_rate": [8.0],
                "hpi": [110.0],  # target column, should not appear in features
            }
        )
        encoded = encode_features(df)
        assert "hpi" not in encoded.columns
