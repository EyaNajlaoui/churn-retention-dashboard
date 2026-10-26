"""Step 2: load and clean the Telco dataset, and build features.

Run:  python -m src.preprocess
"""

import pandas as pd

from src.config import CLEAN_DATA, ID_COL, RAW_DATA, TARGET


def load_raw() -> pd.DataFrame:
    """Load the original Kaggle CSV."""
    return pd.read_csv(RAW_DATA)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data issues in the Telco dataset."""
    df = df.copy()

    # TotalCharges is stored as text and is blank for brand-new customers (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Target as 0/1
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    # TODO: check for duplicates on ID_COL and any other inconsistency found in notebook 01
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Business-meaningful features. Add your own ideas from the exploration."""
    df = df.copy()

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1) + (df["InternetService"] != "No").astype(int)

    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )

    # TODO: other ideas: average charge per month of tenure, has long-term contract, auto-payment flag...
    return df


def main() -> None:
    df = add_features(clean(load_raw()))
    CLEAN_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA, index=False)
    print(f"Saved {len(df):,} rows to {CLEAN_DATA}")
    print(f"Churn rate: {df[TARGET].mean():.1%}  |  unique customers: {df[ID_COL].nunique():,}")


if __name__ == "__main__":
    main()
