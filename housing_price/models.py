import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor

# alphas the RidgeCV/LassoCV search over to pick their regularisation strength
_RIDGE_ALPHAS = np.logspace(-1, 3, 30)
_LASSO_ALPHAS = np.logspace(-4, 0, 30)


def _pipe(estimator):
    # SalePrice is right-skewed, so train on log(price) and invert on predict.
    # Metrics still come out in dollars.
    pipe = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def get_models():
    rs = config.RANDOM_STATE
    return {
        "Ridge Regression": _pipe(RidgeCV(alphas=_RIDGE_ALPHAS, gcv_mode="eigen")),
        "Lasso Regression": _pipe(LassoCV(alphas=_LASSO_ALPHAS, max_iter=20000, random_state=rs)),
        "Random Forest": _pipe(
            RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1)
        ),
        "Gradient Boosting": _pipe(GradientBoostingRegressor(random_state=rs)),
    }
