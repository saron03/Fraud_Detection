from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Example: XGBoost fine-tuning
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
param_grid = {
    "max_depth": [3,5,7],
    "n_estimators": [100,200],
    "learning_rate": [0.01,0.1]
}
grid = GridSearchCV(xgb, param_grid, scoring="roc_auc", cv=3)
grid.fit(X_train_res, y_train_res)
best_xgb = grid.best_estimator_

# Random Forest test
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
rf.fit(X_train_res, y_train_res)
