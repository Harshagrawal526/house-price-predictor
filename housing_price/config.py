"""Paths and constants used across the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA = DATA_DIR / "AmesHousing.csv"
RESULTS_CSV = DATA_DIR / "Model_Evaluation_Results.csv"
RESULTS_MD = REPORTS_DIR / "results.md"
MODEL_PATH = PROJECT_ROOT / "models" / "housing_model.pkl"

TARGET = "SalePrice"
DROP_COLS = ["Id"]  # not predictive
CURRENCY = "$"

# feature columns are detected from dtypes at runtime (numeric vs categorical),
# so the 79 Ames columns don't need to be listed by hand

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
