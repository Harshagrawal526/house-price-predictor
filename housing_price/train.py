import joblib

from . import config
from .data import load_data, split_features_target
from .evaluate import cross_validate_models
from .models import get_models


def train_and_save(path=config.MODEL_PATH):
    # pick the best model by CV, then refit it on the whole dataset and save
    df = load_data()
    X, y = split_features_target(df)
    results = cross_validate_models(get_models(), X, y)
    best_name = results.iloc[0]["Model"]

    model = get_models()[best_name]
    model.fit(X, y)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "name": best_name}, path)
    return path


def load_model(path=config.MODEL_PATH):
    if not path.exists():
        train_and_save(path)
    return joblib.load(path)
