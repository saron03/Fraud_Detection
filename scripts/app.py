import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from src.fraud_detection.model_training import Trainer

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
