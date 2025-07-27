# Task 2 - Model Building and Evaluation

This phase focuses on building and evaluating machine learning models to detect fraudulent transactions in two datasets:

1. **E-commerce Fraud Dataset (`Fraud_Data.csv`)**
2. **Bank Credit Card Dataset (`creditcard.csv`)**

---

## ✅ Objectives

- Prepare train/test data with preprocessing
- Handle class imbalance using **SMOTE**
- Train two models: **Logistic Regression** and **XGBoost**
- Evaluate using metrics like:
  - **F1-Score**
  - **AUC-ROC**
  - **AUC-PR (Precision-Recall AUC)**
- Compare and justify best-performing model

---

##  Models Trained

| Dataset     | Model               | Highlights                          |
|-------------|---------------------|-------------------------------------|
| E-commerce  | Logistic Regression | Simple, explainable                 |
|             | XGBoost             | High F1 and PR AUC, robust to noise |
| Credit Card | Logistic Regression | Performs well with class weights    |
|             | XGBoost             | Strong results with class imbalance |

---

##  Evaluation Summary

- **XGBoost** outperformed Logistic Regression in both datasets.
- **AUC-PR** was the key metric due to imbalanced classes.
- **Recommendation**: Use **XGBoost** for deployment, with SHAP for explainability.

---

## Output Files

- `notebooks/05_model_building.ipynb`: Full modeling and evaluation code
- `reports/figures/`: ROC and PR Curve plots
-  `reports/Task-2_Report.md`: Summary of model comparison

---

##  Next Step

Use SHAP and other interpretability tools to explain model predictions (Task 3).

