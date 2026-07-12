import numpy as np

from housing_price import config
from housing_price.data import load_data, split_features_target, train_test_data
from housing_price.evaluate import cross_validate_models, evaluate_on_test
from housing_price.models import get_models


def test_data_loads_without_nulls():
    df = load_data()
    assert len(df) == 545
    assert df.isnull().sum().sum() == 0
    assert config.TARGET in df.columns


def test_target_is_not_scaled():
    df = load_data()
    assert df[config.TARGET].min() > 100_000


def test_models_produce_sensible_scores():
    df = load_data()
    X, y = split_features_target(df)
    results = cross_validate_models(get_models(), X, y)
    assert np.isfinite(results["R2"]).all()
    assert results["R2"].max() > 0.5


def test_held_out_evaluation_runs():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_data(df)
    best = get_models()["Random Forest"]
    metrics = evaluate_on_test(best, X_train, y_train, X_test, y_test)
    assert 0 < metrics["r2"] <= 1
    assert metrics["rmse"] > 0
    assert len(metrics["y_pred"]) == len(y_test)
