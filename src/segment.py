"""K-Means customer segmentation helpers (explored in notebook 05)."""

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import N_SEGMENTS, RANDOM_STATE, SEGMENT_FEATURES, SEGMENT_PATH


def fit_segments(df: pd.DataFrame, k: int = N_SEGMENTS):
    scaler = StandardScaler().fit(df[SEGMENT_FEATURES])
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(scaler.transform(df[SEGMENT_FEATURES]))
    return scaler, kmeans


def default_names(df: pd.DataFrame, labels) -> dict:
    """Readable names from each cluster's profile (rename them in notebook 05 if you prefer)."""
    profile = df.assign(cluster=labels).groupby("cluster")[SEGMENT_FEATURES].mean()
    t_med, m_med = df["tenure"].median(), df["MonthlyCharges"].median()
    names = {}
    for c, row in profile.iterrows():
        name = ("Loyal" if row["tenure"] >= t_med else "New") + (", high bill" if row["MonthlyCharges"] >= m_med else ", low bill")
        names[c] = name if name not in names.values() else f"{name} ({c})"
    return names


def save_segments(scaler, kmeans, names: dict) -> None:
    SEGMENT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "kmeans": kmeans, "features": SEGMENT_FEATURES, "names": names}, SEGMENT_PATH)


def assign_segments(df: pd.DataFrame) -> pd.Series:
    bundle = joblib.load(SEGMENT_PATH)
    labels = bundle["kmeans"].predict(bundle["scaler"].transform(df[bundle["features"]]))
    return pd.Series(labels, index=df.index).map(bundle["names"]).rename("Segment")
