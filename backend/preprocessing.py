"""Row validation and feature encoding for the HPI prediction model.

Target: hpi (house price index), predicted from year, hicp, and
unemployment_rate, with country as a categorical feature.
"""
from typing import Any, Dict

import pandas as pd

VALID_YEAR_RANGE = (2000, 2035)
REQUIRED_FEATURE_COLUMNS = ["country", "year", "hicp", "unemployment_rate"]


def validate_row(row: Dict[str, Any]) -> None:
    """Validate a single input row before training or prediction.

    Args:
        row: A dict with keys country, year, hicp, unemployment_rate.

    Raises:
        ValueError: if any field is missing, the wrong type, or out of a
            plausible range.
    """
    for field in REQUIRED_FEATURE_COLUMNS:
        if field not in row or row[field] is None:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(row["country"], str) or len(row["country"]) != 2:
        raise ValueError("country must be a 2-letter ISO code string")

    try:
        year = int(row["year"])
    except (TypeError, ValueError):
        raise ValueError("year must be an integer")
    if not (VALID_YEAR_RANGE[0] <= year <= VALID_YEAR_RANGE[1]):
        raise ValueError(
            f"year must be between {VALID_YEAR_RANGE[0]} and {VALID_YEAR_RANGE[1]}"
        )

    for field in ["hicp", "unemployment_rate"]:
        try:
            float(row[field])
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be numeric")

    if not (-50 <= float(row["unemployment_rate"]) <= 100):
        raise ValueError("unemployment_rate must be a plausible percentage")


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode `country` and return a numeric feature frame.

    Keeps year, hicp, unemployment_rate as-is and adds one binary column
    per country (e.g. country_AT, country_BE, ...).

    Args:
        df: DataFrame containing at least REQUIRED_FEATURE_COLUMNS.

    Returns:
        A new DataFrame ready to feed into model training.

    Raises:
        ValueError: if required columns are missing.
    """
    missing = set(REQUIRED_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Cannot encode, missing columns: {missing}")

    encoded = pd.get_dummies(
        df[REQUIRED_FEATURE_COLUMNS], columns=["country"], prefix="country"
    )
    return encoded
