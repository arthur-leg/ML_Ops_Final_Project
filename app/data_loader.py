"""Download and load the HPI dataset from Google Drive.

The raw data is stored on Google Drive and versioned locally with DVC.
This module only handles getting the CSV onto disk and into a DataFrame
with a validated schema -- it does not do any feature engineering.
"""
import os
from types import SimpleNamespace

import pandas as pd

try:
    import gdown
except ImportError:
    gdown = SimpleNamespace(
        download=lambda *args, **kwargs: (_ for _ in ()).throw(
            ModuleNotFoundError("No module named 'gdown'")
        )
    )

REQUIRED_COLUMNS = ["country", "year", "hpi", "hicp", "unemployment_rate"]


def download_csv(file_id: str, output_path: str = "data/raw_data.csv") -> str:
    """Download the dataset CSV from Google Drive using its file ID.

    Args:
        file_id: The Google Drive file ID (the id= value from the shareable link).
        output_path: Local path to save the downloaded CSV.

    Returns:
        The path to the downloaded file.
    """
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=True)
    return output_path


def load_data(csv_path: str) -> pd.DataFrame:
    """Load the dataset from disk and validate its schema.

    Args:
        csv_path: Path to a local CSV file.

    Returns:
        A DataFrame with at least the REQUIRED_COLUMNS present.

    Raises:
        FileNotFoundError: if csv_path doesn't exist.
        ValueError: if required columns are missing from the CSV.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No data file found at {csv_path}")

    df = pd.read_csv(csv_path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return df
