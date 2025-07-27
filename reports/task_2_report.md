# Model Comparison

## 1. Logistic Regression (E-commerce)

- F1 Score: 0.27

- AUC-PR: 0.39

## 2. XGBoost (E-commerce)

- F1 Score: 0.68

- AUC-PR: 0.60


## 3. Logistic Regression (Bank)

- F1 Score: 0.11

- AUC-PR: 0.77

## 4. XGBoost (Bank)

- F1 Score: 0.63

- AUC-PR: 0.85

# Insights

- Across both datasets, XGBoost significantly outperformed Logistic Regression in terms of F1 score and AUC-PR.

- For fraud detection, AUC-PR is especially important due to class imbalance, and XGBoost showed much higher scores.

- Logistic Regression is more interpretable and easier to explain, but its performance was poor (especially F1 score of 0.11 for bank fraud).

# Recommendation

I recommend deploying XGBoost for both datasets due to its superior fraud detection capability.