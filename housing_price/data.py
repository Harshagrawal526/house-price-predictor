"""Data loading and train/test splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_data(path: str | Path = config.RAW_DATA) -> pd.DataFrame:
    """Load the raw housing dataset.

    Parameters
    ----------
    path:
        Location of the CSV file. Defaults to ``data/Housing.csv``.

    Returns
    -------
    pandas.DataFrame
        The raw dataset with all original columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Expected the Kaggle Housing CSV."
        )
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame):
    """Split a dataframe into the feature matrix ``X`` and target vector ``y``."""
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].copy()
    return X, y


def train_test_data(df: pd.DataFrame):
    """Return a reproducible train/test split of features and target."""
    X, y = split_features_target(df)
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )
