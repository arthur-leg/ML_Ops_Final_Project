import pandas as pd
import pytest

from app.preprocessing import encode_features, validate_row

pytestmark = pytest.mark.unit


class TestValidateRow:
    def test_valid_row_passes(self):
        row = {"country": "AT", "year": 2015, "hicp": 0.8, "unemployment_rate": 6.1}
        validate_row(row)  # should not raise

    def test_missing_field_raises(self):
        row = {"country": "AT", "year": 2015, "hicp": 0.8}
        with pytest.raises(ValueError, match="Missing required field"):
            validate_row(row)

    def test_invalid_country_code_raises(self):
        row = {"country": "Austria", "year": 2015, "hicp": 0.8, "unemployment_rate": 6.1}
        with pytest.raises(ValueError, match="country"):
            validate_row(row)

    def test_out_of_range_year_raises(self):
        row = {"country": "AT", "year": 1800, "hicp": 0.8, "unemployment_rate": 6.1}
        with pytest.raises(ValueError, match="year"):
            validate_row(row)

    def test_non_numeric_unemployment_raises(self):
        row = {"country": "AT", "year": 2015, "hicp": 0.8, "unemployment_rate": "high"}
        with pytest.raises(ValueError, match="unemployment_rate"):
            validate_row(row)


class TestEncodeFeatures:
    def test_encode_creates_country_dummies(self):
        df = pd.DataFrame(
            [
                {"country": "AT", "year": 2015, "hpi": 100.0, "hicp": 0.8, "unemployment_rate": 6.1},
                {"country": "BE", "year": 2015, "hpi": 100.0, "hicp": 0.6, "unemployment_rate": 8.7},
            ]
        )
        encoded = encode_features(df)
        assert "country_AT" in encoded.columns
        assert "country_BE" in encoded.columns
        assert encoded.loc[0, "country_AT"] == 1
        assert encoded.loc[0, "country_BE"] == 0

    def test_encode_missing_column_raises(self):
        df = pd.DataFrame([{"country": "AT", "year": 2015}])
        with pytest.raises(ValueError, match="missing columns"):
            encode_features(df)
