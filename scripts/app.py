import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from src.model_training import Trainer

st.title("Fraud Detection Dashboard")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    st.subheader("Model Predictions")
    trainer = Trainer()
    # Assuming pipeline saved X_train/X_test and y_train/y_test as .npy
    X_test = np.load("data/processed/X_test.npy")
    # Load pre-trained model (example)
    lr_model = joblib.load("models/logreg_ecom.pkl")
    y_pred = lr_model.predict(X_test)
    st.write(y_pred)

    st.subheader("SHAP Feature Importance")
    explainer = shap.Explainer(lr_model, X_test)
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values)
    st.pyplot(bbox_inches="tight")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from lime.lime_tabular import LimeTabularExplainer
from src.fraud_detection.model_training import Trainer

st.title("Fraud Detection Dashboard")

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data")
    st.dataframe(df.head())

    st.subheader("Model Predictions")
    X_test = np.load("data/processed/X_test.npy")
    lr_model = joblib.load("models/logreg_ecom.pkl")  # saved trained model
    y_pred = lr_model.predict(X_test)
    y_prob = lr_model.predict_proba(X_test)[:,1]
    st.write(pd.DataFrame({"Prediction": y_pred, "Fraud Probability": y_prob}))

    st.subheader("SHAP Feature Importance")
    explainer = shap.Explainer(lr_model, X_test)
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(bbox_inches="tight")

    st.subheader("LIME Local Explanation")
    lime_exp = LimeTabularExplainer(
        training_data=np.array(X_test),
        feature_names=[f"f{i}" for i in range(X_test.shape[1])],
        class_names=["Not Fraud", "Fraud"],
        mode="classification"
    )
    i = st.slider("Select instance index for LIME explanation", 0, len(X_test)-1, 0)
    exp = lime_exp.explain_instance(X_test[i], lr_model.predict_proba, num_features=10)
    st.write(exp.as_list())
