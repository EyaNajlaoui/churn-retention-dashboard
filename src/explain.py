"""SHAP helpers: global drivers and the main risk reason for each customer.

One-hot columns (e.g. Contract_Month-to-month, Contract_Two year) are summed back
to their original column (Contract), so a reason reads like
"Contract: Month-to-month" instead of a technical feature name.
"""

import numpy as np
import pandas as pd
import shap

READABLE = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "NumServices": "Number of services",
    "SeniorCitizen": "Senior citizen",
    "PaymentMethod": "Payment",
    "InternetService": "Internet",
    "TechSupport": "Tech support",
    "OnlineSecurity": "Online security",
    "PaperlessBilling": "Paperless billing",
    "TenureGroup": "Tenure group",
}


def shap_by_original_column(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (one row per customer) aggregated to the original columns."""
    prep, model = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    X_enc = prep.transform(X)
    names = prep.get_feature_names_out()  # e.g. "cat__Contract_One year", "num__tenure"

    values = shap.TreeExplainer(model).shap_values(X_enc)
    if isinstance(values, list):  # some SHAP versions return one array per class
        values = values[1]

    cat_cols = prep.transformers_[0][2]
    original = []
    for n in names:
        kind, rest = n.split("__", 1)
        if kind == "cat":
            original.append(next(c for c in sorted(cat_cols, key=len, reverse=True) if rest.startswith(c + "_")))
        else:
            original.append(rest)

    shap_df = pd.DataFrame(values, columns=names, index=X.index)
    return shap_df.T.groupby(np.array(original)).sum().T


def global_importance(shap_cols: pd.DataFrame, top: int = 10) -> pd.Series:
    """Mean |SHAP| per original column, highest first."""
    return shap_cols.abs().mean().sort_values(ascending=False).head(top)


def _label(col: str, value) -> str:
    name = READABLE.get(col, col)
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return f"{name}: {value:g}"
    return f"{name}: {value}"


def top_reasons(shap_cols: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """For each customer, the column that pushes churn risk up the most, with the customer's value."""
    top_col = shap_cols.idxmax(axis=1)
    return pd.Series([_label(c, X.at[i, c]) for i, c in top_col.items()], index=X.index, name="TopRiskReason")
