import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def build_preprocessor():
    # numeric: fill missing with the median, scale, then clip to +/-3 SD. Ames
    # has extreme outliers (e.g. a 215k sqft lot); clipping bounds their
    # influence so the linear models can't extrapolate to absurd predictions.
    # categorical: fill missing with "missing" (often means "none" in Ames),
    # then one-hot encode. Columns are picked by dtype so all 79 are handled.
    # Wrapped in a ColumnTransformer so it's refit per fold and doesn't leak.
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clip", FunctionTransformer(
            np.clip, kw_args={"a_min": -3, "a_max": 3}, feature_names_out="one-to-one"
        )),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        # min_frequency groups rare levels together, which keeps the linear
        # models from blowing up on the ~260 one-hot columns
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            # dtype_exclude="number" catches every non-numeric column regardless
            # of whether strings are numpy-object or pyarrow-backed (pandas 3.x)
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ]
    )
