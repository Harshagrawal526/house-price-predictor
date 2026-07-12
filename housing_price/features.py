"""Feature preprocessing.

The preprocessing lives inside a scikit-learn ``ColumnTransformer`` so that it
is *fitted only on the training portion of each cross-validation fold*. This is
the key fix over the original project, which scaled the whole dataset up front
and leaked information from the test rows into training.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Build the feature preprocessing transformer.

    - Numeric columns are standardised (zero mean, unit variance).
    - Categorical columns are one-hot encoded; unknown categories seen at
      prediction time are ignored rather than raising.

    Note the target (``price``) is intentionally *not* transformed here, so all
    error metrics stay in real rupees and remain interpretable.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
