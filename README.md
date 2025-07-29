# Task 3 - Model Explainability (SHAP)

This task uses **SHAP (Shapley Additive exPlanations)** to interpret our best-performing fraud detection model.

## Steps Performed
1. **Trained Best Model** – Selected the highest-performing model from previous experiments.
2. **Applied SHAP** – Used SHAP to compute feature contributions for each prediction.
3. **Generated Plots**:
   - **Summary Plot** – Shows overall feature importance and their impact on predictions.
   - **Force Plot** – Explains individual predictions (local interpretability).

## Key Insights
- **Global Interpretation (Summary Plot):**  
  The top features influencing fraud classification were:  
  - `Feature A` – Strongly increases fraud probability.  
  - `Feature B` – Has a mixed effect depending on its value.  
  - `Feature C` – Consistently lowers fraud likelihood.  

- **Local Interpretation (Force Plot):**  
  Force plots helped us understand why certain transactions were flagged as fraud by visualizing the feature contributions for specific predictions.

## Conclusion
Using SHAP allowed us to **identify key drivers of fraud** and **understand individual predictions**, improving both transparency and trust in the model.
