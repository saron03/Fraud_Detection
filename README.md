# Fraud Detection for E-Commerce and Banking  
**Adey Innovations Inc.**

---

## Overview

This project aims to build accurate and interpretable fraud detection models for **e-commerce** and **bank credit transactions**. Fraud detection is a mission-critical task for financial technology companies. The challenge is to detect fraudulent transactions with minimal false positives to maintain a secure and frictionless user experience.

---

## Project Structure

```plaintext
.
├── data/
│   ├── raw/
│   │   ├── Fraud_Data.csv
│   │   ├── IpAddress_to_Country.csv
│   │   ├── creditcard.csv
│   ├── processed/
│   │   ├── clean_Fraud_Data.csv
│   │   ├── merged_Fraud_Data.csv
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_data_transformation.ipynb
│   ├── 05_model_building.ipynb
│   ├── 06_model_explainability.ipynb
│
├── reports/
│   ├── Task-1_Report.md
│   ├── Final_Report.pdf
│
├── scripts/
│   ├── utils.py
│   ├── model_pipeline.py
│
├── requirements.txt
├── README.md

```

## Project Goals

- Analyze, clean, and preprocess large transactional datasets.

- Engineer new features (e.g., time_since_signup, geolocation mapping).

- Handle severe class imbalance with SMOTE and smart sampling.

- Train and compare Logistic Regression and Ensemble Models (XGBoost).

- Evaluate models using fraud-sensitive metrics (F1, ROC AUC, AUC-PR).

- Explain predictions with SHAP to reveal fraud drivers.

- Deliver a clear, reproducible workflow with actionable insights.

## Datasets

- Fraud_Data.csv — E-commerce transactions.

- IpAddress_to_Country.csv — IP-to-country mapping for geolocation.

- creditcard.csv — Bank credit transactions (from PCA).

## Key Steps

1. Data Cleaning & Preprocessing

- Removed duplicates, handled missing values, corrected datatypes.
- Cleaned timestamps, IP addresses converted to integers for merging.
- Saved cleaned data for reproducibility.

2. Exploratory Data Analysis (EDA)

- Verified class imbalance (fraud: ~0.5%–1%).
- Identified fraud trends: more fraud at night, younger users slightly higher fraud, older browsers riskier.
- Mapped IP addresses to countries for location-based risk.

3. Feature Engineering

- Created time_since_signup (purchase_time – signup_time).
- Added hour_of_day and day_of_week features.
- Merged IP mapping to assign each transaction a country.

4. Handling Class Imbalance

- Applied SMOTE to balance training data:
```
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```
- Used class weighting during model training.
- Focused on recall, F1, AUC-PR, not just accuracy.

5. Model Building & Evaluation

- Baseline: Logistic Regression.
- Advanced: XGBoost Ensemble.
- Evaluate with confusion matrix, ROC AUC, precision-recall curve.

6. Model Explainability (SHAP)

- Use SHAP to interpret global & local drivers of fraud.
- Generate summary plots & force plots to communicate results.

## How to Run

1. Clone Repo

```
git clone https://github.com/saron03/Fraud_Detection.git  
cd Fraud_Detection
```

2. Create Virtual Environment

```
python3 -m venv venv  
source venv/bin/activate
```

3. Install Requirements

```
pip install -r requirements.txt
```

4. Run Notebooks

Open notebooks with Jupyter or VS Code and follow the pipeline step by step.

## Final Outputs

- Interim Reports: Detailed analysis, EDA, feature engineering, imbalance handling.

- Models: Fully trained baseline and ensemble models.

- Explainability: SHAP plots with insights into fraud drivers.

- Professional Documentation: Final report PDF and this README.