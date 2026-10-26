"""Step 6: score every customer and export the table used by Power BI.

Run:  python -m src.score
Prerequisites: python -m src.preprocess, python -m src.train, and notebook 05 (segmentation)

Output: data/processed/customers_scored.csv with all original columns plus
    ChurnProbability, PredictedChurn, RiskLevel, Segment, TopRiskReason, RevenueAtRisk
"""

import joblib
import numpy as np
import pandas as pd

from src.config import CLEAN_DATA, HIGH_RISK, ID_COL, MODEL_PATH, SCORED_DATA, SEGMENT_PATH, TARGET
from src.explain import shap_by_original_column, top_reasons
from src.segment import assign_segments


def main() -> None:
    df = pd.read_csv(CLEAN_DATA)
    bundle = joblib.load(MODEL_PATH)
    model, threshold = bundle["model"], bundle["threshold"]
    X = df.drop(columns=[TARGET, ID_COL])

    df["ChurnProbability"] = model.predict_proba(X)[:, 1].round(4)
    df["PredictedChurn"] = (df["ChurnProbability"] >= threshold).astype(int)
    # Medium + High = customers on the contact list (PredictedChurn = 1)
    df["RiskLevel"] = np.select(
        [df["ChurnProbability"] >= HIGH_RISK, df["ChurnProbability"] >= threshold], ["High", "Medium"], default="Low")
    df["RevenueAtRisk"] = (df["MonthlyCharges"] * df["ChurnProbability"]).round(2)  # expected monthly loss

    df["TopRiskReason"] = top_reasons(shap_by_original_column(model, X), X)

    if SEGMENT_PATH.exists():
        df["Segment"] = assign_segments(df)
    else:
        print("Segmentation not found: run notebook 05 first. Segment set to 'Not segmented'.")
        df["Segment"] = "Not segmented"

    # Readable target for Power BI
    df["ChurnLabel"] = df[TARGET].map({1: "Churned", 0: "Active"})

    SCORED_DATA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORED_DATA, index=False)
    print(f"Saved {len(df):,} scored customers to {SCORED_DATA}")
    print(df["RiskLevel"].value_counts().to_string())
    print(f"Monthly revenue at risk: ${df['RevenueAtRisk'].sum():,.0f}")
    # Sanity check: with realistic probabilities, both numbers should be close
    print(f"Expected churners (sum of probabilities): {df['ChurnProbability'].sum():,.0f} | actual churners: {int(df[TARGET].sum()):,}")


if __name__ == "__main__":
    main()
