# Customer Churn Prediction & Retention Dashboard

> Predict which telecom customers are likely to leave, explain why, and give the retention team a prioritized action list in Power BI.

![Python](https://img.shields.io/badge/Python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-explainability-941B50?style=flat-square)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat-square&logoColor=black)

<!-- TODO: add a screenshot of the dashboard and the public link once it is published -->
<!-- ![Dashboard](reports/figures/dashboard_overview.png) -->
<!-- **Live dashboard:** [Power BI report](https://app.powerbi.com/view?r=YOUR-LINK) -->

## Business problem

Keeping a customer costs much less than acquiring a new one, yet churn is usually noticed only after it happens.
This project answers three questions a retention manager asks every week:

1. **Who** is likely to leave?
2. **How much revenue** is at risk?
3. **Who should we contact first**, and with which offer?

## Key results

| Metric | Value |
|---|---|
| ROC-AUC (test set) | **0.848** |
| Churners detected (recall) | **81.6%** (305 / 374) |
| Precision of the contact list | **52%** (vs. 26.5% if customers were picked at random) |
| Decision threshold | 0.25 |
| Monthly revenue at risk (all customers) | **$138.8K** |
| Calibration check | 1,869 expected churners vs. 1,869 actual |

The model was tuned to maximize **precision at 80% recall**: missing a churner (a lost customer) costs far more
than calling a customer who would have stayed (a call or a small discount).

## Main insights

- **Commitment is the strongest signal.** Customers without a long-term contract churn at **42.7%**, versus **6.8%**
  for one- or two-year contracts. For the riskiest customers, SHAP identifies the month-to-month contract as the main risk reason.
- **The first months are critical.** The median tenure of churners is **10 months**, versus **38 months** for active customers.
- **Churners pay more.** Median monthly charge of **$79.65** for churners vs. **$64.43** for active customers.
- **Automatic payment and protection services retain customers.** Churn is **16%** with automatic payment vs. **34.7%**
  without, and drops from **29.8%** (no protection service) to **5.3%** (four protection services).

### Customer segments (K-Means, k = 4)

| Segment | Customers | Avg tenure | Avg monthly | Churn rate | High-risk customers | Recommended action |
|---|---|---|---|---|---|---|
| **New, high bill** | 2,410 | 15 months | $77 | **47%** | **1,162 (78% of all high-risk)** | Discount or free month in exchange for a 1-year contract |
| New, low bill | 1,509 | 10 months | $29 | 24% | 224 | Upsell protection services (security, tech support) |
| Loyal, high bill | 2,152 | 57 months | $91 | 16% | 102 | Protect with loyalty rewards, no costly discounts |
| Loyal, low bill | 972 | 54 months | $30 | 4% | 0 | No action needed |

k = 4 was chosen with the elbow method and the best silhouette score (0.425).

## Approach

| Step | Notebook / script | What was done |
|---|---|---|
| 1. Exploration | `01_exploration` | Data quality, churn rate by category, numeric distributions |
| 2. Preprocessing | `02_preprocessing`, `src/preprocess.py` | Cleaning (`TotalCharges`), feature engineering, test of extra features |
| 3. Modeling | `03_modeling`, `src/train.py` | Logistic Regression vs. Random Forest vs. XGBoost (5-fold CV), class weights vs. SMOTE, calibration, threshold choice |
| 4. Tuning | `03b_tuning` | Randomized search on 9 XGBoost hyperparameters, optimized for precision at 80% recall |
| 5. Explainability | `04_explainability`, `src/explain.py` | SHAP global drivers and main risk reason per customer |
| 6. Segmentation | `05_segmentation`, `src/segment.py` | K-Means segments, profiling and business actions |
| 7. Scoring | `src/score.py` | One row per customer for Power BI: probability, risk level, segment, reason, revenue at risk |
| 8. Dashboard | `powerbi/` | Power BI report (see below) |

### Modeling decisions

- **Model choice:** the three models scored within 0.001 ROC-AUC (≈ 0.845 in cross-validation). XGBoost was kept
  for its non-linear effects and fast, exact SHAP explanations.
- **Imbalance (26.5% churn):** class weights and SMOTE gave the same ROC-AUC (0.845). But both **inflate the
  probabilities**: with class weights, the model expected 555 churners on the test set instead of 374. Since the
  dashboard turns probabilities into dollars, the final model is trained **without weights** (well calibrated:
  374.2 expected vs. 374 actual), and recall is recovered by **lowering the decision threshold**.
- **Tuning:** precision at 80% recall went from 51.5% to 52.2% on the test set (ROC-AUC 0.844 → 0.848). The gain is
  smaller than the variation between cross-validation folds (std 0.02), and extra engineered features did not help:
  **performance is limited by the available data rather than by the model**.

## Dashboard (Power BI)

| Page | Content |
|---|---|
| Executive overview | Churn rate, revenue at risk, high-risk customers, churn by contract and tenure |
| Segments & drivers | Segment size and risk, main churn drivers (SHAP) |
| Action list | Customers to contact, sorted by revenue at risk, with their main risk reason |
| What-if | "% of at-risk customers retained" slider → monthly and annual revenue saved |

Source table: `data/processed/customers_scored.csv` (generated by `src/score.py`).

## Project structure

```
├── data/
│   ├── raw/                 original Kaggle CSV (not versioned)
│   └── processed/           cleaned data + customers_scored.csv (not versioned, regenerated by the scripts)
├── notebooks/               01 exploration → 05 segmentation (with outputs)
├── src/
│   ├── config.py            paths and settings
│   ├── preprocess.py        cleaning and features
│   ├── train.py             final model training
│   ├── explain.py           SHAP helpers
│   ├── segment.py           K-Means helpers
│   └── score.py             scoring table for Power BI
├── models/                  xgb_best_params.json (tuned hyperparameters); trained models are not versioned
├── powerbi/                 Power BI report (.pbix)
└── reports/
    ├── figures/             charts used in this README
    ├── model_comparison.csv
    └── shap_importance.csv
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

1. Download [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and put
   `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/raw/`.
2. Run the pipeline:

```bash
python -m src.preprocess     # clean data + features
python -m src.train          # train the final model (uses models/xgb_best_params.json)
```

3. Run notebook `05_segmentation` once (it saves the segmentation), then:

```bash
python -m src.score          # writes data/processed/customers_scored.csv for Power BI
```

## Limitations and next steps

- **Snapshot data with no dates:** no trend over time; tenure groups are used instead.
- **Limited behavioral information:** no complaints, support calls or usage data. These would be the main lever to
  raise precision above ~52% at 80% recall.
- **Fictional telecom data (IBM sample):** the approach transfers to real CRM data, the figures do not.
- Next steps: estimate the cost of a retention offer to choose the threshold by expected profit, and schedule
  monthly retraining with new customers.

## Tech stack

Python · pandas · scikit-learn · XGBoost · imbalanced-learn · SHAP · matplotlib · seaborn · Power BI · DAX

---

**Author:** Eya Najlaoui · [LinkedIn](https://www.linkedin.com/in/eya-najlaoui/) · [GitHub](https://github.com/EyaNajlaoui)
