from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor():
    # scale numeric features, one-hot encode categoricals. Wrapped in a
    # ColumnTransformer so it's refit per fold and doesn't leak. The target is
    # left alone so metrics stay in rupees.
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
