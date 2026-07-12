from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_data(path=config.RAW_DATA):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at '{path}'")
    return pd.read_csv(path)


def split_features_target(df):
    X = df[config.FEATURES].copy()
    y = df[config.TARGET].copy()
    return X, y


def train_test_data(df):
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
