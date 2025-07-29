# SHAP Summary — E-commerce Fraud Detection (XGBoost)
SHAP analysis revealed that the following features most strongly influence fraud predictions in our XGBoost model:

- Feature 183 had the largest overall impact on predictions, indicating it plays a critical role in detecting fraud.

- Features 2 and 3 also showed high SHAP values, contributing strongly to the model’s decision-making process.

- Features 1, 8, and 0 demonstrated moderate influence, further supporting the prediction logic.

- Other features like 97, 9, 7, and 184 had smaller but still meaningful contributions.

The `SHAP bar plot` illustrated the average magnitude of each feature's influence on the model, while the `beeswarm plot` showed how individual feature values push predictions higher (toward fraud) or lower (toward non-fraud):

- Red values (higher feature values) tend to increase the probability of fraud in some features.

- Blue values (lower feature values) have the opposite effect in others.

- This allows me to see not just which features matter, but how they influence individual predictions.

- These insights gives me confidence in the model's behavior and help identify the patterns the model uses to flag suspicious transactions.