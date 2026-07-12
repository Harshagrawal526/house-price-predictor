import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate

from . import config

# sklearn returns error metrics negated ("higher is better"), so we flip them below
_SCORING = {
    "r2": "r2",
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
}


def cross_validate_models(models, X, y):
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=_SCORING, n_jobs=-1)
        rows.append(
            {
                "Model": name,
                "R2": scores["test_r2"].mean(),
                "R2_std": scores["test_r2"].std(),
                "RMSE": -scores["test_rmse"].mean(),
                "RMSE_std": scores["test_rmse"].std(),
                "MAE": -scores["test_mae"].mean(),
                "MAE_std": scores["test_mae"].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)


def evaluate_on_test(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "y_pred": y_pred,
    }
