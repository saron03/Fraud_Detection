from src.fraud_detection.pipeline import FraudPipeline

if __name__ == "__main__":
    pipe = FraudPipeline()
    outputs = pipe.run("data/processed/feature_engineered_Fraud_Data.csv")
    print("Pipeline complete. Shapes:")
    print("Train:", outputs["X_train"].shape, outputs["y_train"].shape)
    print("Test:", outputs["X_test"].shape, outputs["y_test"].shape)
