"""Model definitions.

Every model is a scikit-learn ``Pipeline`` that bundles the shared preprocessor
with an estimator, so preprocessing is always refitted per fold and there is no
leakage. Bundling also means a single object can be cross-validated, evaluated
and (if desired) persisted as one unit.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from . import config
from .features import build_preprocessor


def _pipe(estimator, *, polynomial_degree: int | None = None) -> Pipeline:
    """Wrap an estimator with the preprocessor (and optional polynomial step)."""
    steps = [("preprocess", build_preprocessor())]
    if polynomial_degree is not None:
        steps.append(
            ("polynomial", PolynomialFeatures(degree=polynomial_degree, include_bias=False))
        )
    steps.append(("model", estimator))
    return Pipeline(steps)


def get_models() -> dict[str, Pipeline]:
    """Return the candidate models, keyed by a human-readable name.

    The line-up is deliberately chosen to tell a story:

    - **Linear Regression** — the baseline.
    - **Ridge / Lasso** — regularised linear models.
    - **Polynomial (deg 2) + Ridge** — unregularised polynomial features on
      binary dummies are numerically unstable, so the degree-2 term is paired
      with L2 regularisation to keep it well-behaved.
    - **Random Forest / Gradient Boosting** — non-linear ensembles that usually
      win on tabular data like this.
    """
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
        "Gradient Boosting": _pipe(
            GradientBoostingRegressor(random_state=rs)
        ),
    }
