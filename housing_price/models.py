from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from . import config
from .features import build_preprocessor


def _pipe(estimator, polynomial_degree=None):
    steps = [("preprocess", build_preprocessor())]
    if polynomial_degree is not None:
        steps.append(
            ("polynomial", PolynomialFeatures(degree=polynomial_degree, include_bias=False))
        )
    steps.append(("model", estimator))
    return Pipeline(steps)


def get_models():
    # Ridge is used on the polynomial features because plain LinearRegression
    # blows up on the degree-2 dummy interactions.
    rs = config.RANDOM_STATE
    return {
        "Linear Regression": _pipe(LinearRegression()),
        "Ridge Regression": _pipe(Ridge(alpha=1.0, random_state=rs)),
        "Lasso Regression": _pipe(Lasso(alpha=1000.0, random_state=rs, max_iter=10000)),
        "Polynomial (deg 2) + Ridge": _pipe(
            Ridge(alpha=10.0, random_state=rs), polynomial_degree=2
        ),
        "Random Forest": _pipe(
            RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1)
        ),
        "Gradient Boosting": _pipe(GradientBoostingRegressor(random_state=rs)),
    }
