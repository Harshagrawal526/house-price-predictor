import numpy as np

from housing_price import config
from housing_price.data import load_data, split_features_target, train_test_data
from housing_price.evaluate import cross_validate_models, evaluate_on_test
from housing_price.models import get_models


def test_data_loads():
    df = load_data()
    assert len(df) == 1460
    assert config.TARGET in df.columns


def test_preprocessing_handles_missing_values():
    # Ames has missing values; features are dropped/split without error and the
    # pipeline imputes them internally (tested via the model scores below).
    df = load_data()
    X, y = split_features_target(df)
    assert config.TARGET not in X.columns
    assert X.isnull().sum().sum() > 0  # raw features do contain gaps


def test_target_is_not_scaled():
    df = load_data()
    assert df[config.TARGET].min() > 10_000


def test_models_produce_sensible_scores():
    df = load_data()
    X, y = split_features_target(df)
    results = cross_validate_models(get_models(), X, y)
    assert np.isfinite(results["R2"]).all()
    assert results["R2"].max() > 0.8


def test_held_out_evaluation_runs():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_data(df)
    best = get_models()["Random Forest"]
    metrics = evaluate_on_test(best, X_train, y_train, X_test, y_test)
    assert 0 < metrics["r2"] <= 1
    assert metrics["rmse"] > 0
    assert len(metrics["y_pred"]) == len(y_test)
