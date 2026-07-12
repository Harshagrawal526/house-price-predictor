import joblib

from . import config
from .data import load_data, split_features_target
from .models import get_models


def train_and_save(path=config.MODEL_PATH, name=config.DEFAULT_MODEL):
    # fit the production model on the whole dataset and save it. No CV here —
    # the model comparison lives in the pipeline; the app just needs a fitted
    # model, so this stays fast enough to run on a cold start.
    df = load_data()
    X, y = split_features_target(df)

    model = get_models()[name]
    model.fit(X, y)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "name": name}, path)
    return path


def load_model(path=config.MODEL_PATH):
    if not path.exists():
        train_and_save(path)
    return joblib.load(path)
