"""Central paths and settings, so every script and notebook uses the same ones."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEAN_DATA = ROOT / "data" / "processed" / "telco_clean.csv"
SCORED_DATA = ROOT / "data" / "processed" / "customers_scored.csv"  # file loaded into Power BI

MODEL_PATH = ROOT / "models" / "churn_model.joblib"          # {"model": pipeline, "threshold": float}
SEGMENT_PATH = ROOT / "models" / "segmentation.joblib"       # {"scaler", "kmeans", "features", "names"}
BEST_PARAMS_PATH = ROOT / "models" / "xgb_best_params.json"  # written by notebook 03b (hyperparameter tuning)
FIGURES_DIR = ROOT / "reports" / "figures"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Risk levels used in the dashboard:
#   High   = probability >= HIGH_RISK
#   Medium = decision threshold <= probability < HIGH_RISK   (still on the contact list)
#   Low    = probability < decision threshold
HIGH_RISK = 0.50

# Features used for customer segmentation (K-Means)
SEGMENT_FEATURES = ["tenure", "MonthlyCharges", "NumServices"]
N_SEGMENTS = 4
