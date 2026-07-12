"""Paths, column groups and constants used across the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA = DATA_DIR / "Housing.csv"
RESULTS_CSV = DATA_DIR / "Model_Evaluation_Results.csv"
RESULTS_MD = REPORTS_DIR / "results.md"

TARGET = "price"

# numeric columns get scaled, categorical columns get one-hot encoded
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "stories", "parking"]
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

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
