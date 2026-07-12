"""Central configuration: paths, column groups and shared constants.

Keeping these in one place means every module agrees on where data lives and
which columns mean what, so there is a single source of truth to change.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA = DATA_DIR / "Housing.csv"
RESULTS_CSV = DATA_DIR / "Model_Evaluation_Results.csv"
RESULTS_MD = REPORTS_DIR / "results.md"

# --- Columns ---------------------------------------------------------------
TARGET = "price"

# Ordered/continuous numeric predictors — these get scaled.
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]

# Categorical predictors (yes/no flags + furnishing status) — one-hot encoded.
CATEGORICAL_FEATURES = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Reproducibility / evaluation -----------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
