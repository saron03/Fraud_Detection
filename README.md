# Task 1 — Data Analysis & Preprocessing

## Project: Improved Fraud Detection for E-commerce & Banking

This repository is a project to improve fraud detection for e-commerce and bank credit transactions.  
**Task 1** focuses on preparing the e-commerce dataset (`Fraud_Data.csv`) for modeling.

---

## What I did in Task 1

1. **Data Cleaning**
   - Removed duplicates
   - Handled missing values
   - Corrected data types

2. **Exploratory Data Analysis (EDA)**
   - Univariate & bivariate analysis
   - Visualized age distribution and fraud patterns

3. **Geolocation Merge**
   - Converted IP addresses to integer format
   - Merged with `IpAddress_to_Country.csv` to add `country` feature

4. **Feature Engineering**
   - Created `time_since_signup` to capture user behavior
   - Added `hour_of_day` & `day_of_week` from `purchase_time`

5. **Data Transformation**
   - Split data into training/testing sets
   - Handled class imbalance with **SMOTE**
   - Scaled numerical features with `StandardScaler`
   - Encoded categorical variables with one-hot encoding

---

##  Structure

- **notebooks/** — Contains 4 notebooks:  
  - `1_Data_Cleaning.ipynb`
  - `2_EDA.ipynb`
  - `3_Feature_Engineering.ipynb`
  - `4_Data_Transformation.ipynb`
- **reports/** — Contains figures: `age_distribution.png`, `age_vs_class.png`
- **data/** — Raw & processed datasets
- **requirements.txt** — Project dependencies

---

## Next Step

Task 2 will cover **model building, training, and evaluation**.  
Stay tuned!

---
