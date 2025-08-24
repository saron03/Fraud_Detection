import pandas as pd
import numpy as np
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

class Trainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.smote = SMOTE(random_state=random_state)

    def _ecom_preprocessor(self, num_features, cat_features):
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
            ]
        )

    def prepare_ecom(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = df.drop("class", axis=1)
        y = df["class"]

        num_features = ["purchase_value", "age", "time_since_signup", "hour_of_day"]
        cat_features = ["source", "browser", "sex", "country"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        preprocessor = self._ecom_preprocessor(num_features, cat_features)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        X_train_res, y_train_res = self.smote.fit_resample(X_train_transformed, y_train)
        return X_train_res, X_test_transformed, y_train_res, y_test

    def prepare_bank(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X = df.drop("Class", axis=1)
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_res, y_train_res = self.smote.fit_resample(X_train_scaled, y_train)
        return X_train_res, X_test_scaled, y_train_res, y_test

    def train_logreg(self, X_train, y_train) -> LogisticRegression:
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(X_train, y_train)
        return model

    def train_xgb(self, X_train, y_train, scale_pos_weight: float = 1.0) -> XGBClassifier:
        model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.random_state
        )
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X_test, y_test, label: str, fig_path: str = None) -> Dict[str, float]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        auc_pr = auc(recall, precision)

        if fig_path:
            plt.figure()
            plt.plot(recall, precision, label=f"PR curve (AUC = {auc_pr:.2f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall Curve: {label}")
            plt.legend()
            plt.savefig(fig_path)
            plt.close()

        return {"f1": f1, "roc_auc": roc, "auc_pr": auc_pr, "report": report}
