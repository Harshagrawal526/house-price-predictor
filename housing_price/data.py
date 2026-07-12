from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_data(path=config.RAW_DATA):
    path = Path(path)
    if not path.exists():
        _download_dataset(path)
    return _normalize_dtypes(pd.read_csv(path))


def _normalize_dtypes(df):
    # pandas 3.x reads text columns as pyarrow-backed strings, which downstream
    # dtype checks and imputers treat differently. Coerce non-numeric columns to
    # plain object with np.nan for missing, so behaviour matches every pandas
    # version.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            mask = df[col].notna()
            df[col] = df[col].astype(object).where(mask, np.nan)
    return df


def _download_dataset(path):
    # fallback so the app self-heals if the CSV is missing (e.g. a hosted
    # environment that didn't pull the data file); fetches the same Ames dataset
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(name="house_prices", version=1, as_frame=True).frame
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def split_features_target(df):
    X = df.drop(columns=[config.TARGET, *config.DROP_COLS])
    y = df[config.TARGET].copy()
    return X, y


def train_test_data(df):
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
