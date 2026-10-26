"""Step 3: train the churn model and save it.

Run:  python -m src.train
Prerequisite: python -m src.preprocess

The model choice and decision threshold come from notebook 03.
"""

import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import BEST_PARAMS_PATH, CLEAN_DATA, ID_COL, MODEL_PATH, RANDOM_STATE, TARGET, TEST_SIZE

# Decision threshold chosen in notebook 03 (update it with your own result)
THRESHOLD = 0.25


def load_xy(path=CLEAN_DATA):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET, ID_COL])
    y = df[TARGET]
    return X, y


def split(X, y):
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def column_types(X: pd.DataFrame):
    # Everything that is not a number is categorical (works with pandas 2 "object" and pandas 3 "str" dtypes)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return categorical, numeric


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical, numeric = column_types(X)
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numeric),
    ])


def make_model(name: str, y_train: pd.Series, balanced: bool = False):
    """Candidate models compared in notebook 03.

    balanced=True gives churners more weight during training. It does not change the ranking (ROC-AUC) but
    inflates the probabilities, so the final model uses balanced=False and a lower threshold instead
    (see the calibration section of notebook 03).
    """
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if balanced else 1.0  # ~2.8 for Telco
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced" if balanced else None)
    if name == "xgboost":
        params = dict(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                      min_child_weight=3)
        if BEST_PARAMS_PATH.exists():  # tuned values from notebook 03b replace the defaults
            params.update(json.loads(BEST_PARAMS_PATH.read_text()))
        return XGBClassifier(**params, scale_pos_weight=pos_weight, eval_metric="auc", random_state=RANDOM_STATE)
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(X: pd.DataFrame, y_train: pd.Series, model_name: str = "xgboost", balanced: bool = False) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(X)), ("model", make_model(model_name, y_train, balanced))])


def evaluate(pipeline, X_test, y_test, threshold: float = THRESHOLD) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {"roc_auc": roc_auc_score(y_test, proba), "recall_churn": recall_score(y_test, pred)}
    print(f"ROC-AUC: {metrics['roc_auc']:.3f} | Recall (churners) @ {threshold:.2f}: {metrics['recall_churn']:.3f}")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["Stay", "Churn"]))
    return metrics


def main(threshold: float = THRESHOLD, model_name: str = "xgboost") -> None:
    X, y = load_xy()
    X_train, X_test, y_train, y_test = split(X, y)

    pipeline = build_pipeline(X, y_train, model_name)
    pipeline.fit(X_train, y_train)
    evaluate(pipeline, X_test, y_test, threshold)

    # Refit on all data for scoring every customer in the dashboard
    final = build_pipeline(X, y, model_name).fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "threshold": threshold}, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
