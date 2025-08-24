import pandas as pd
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import Trainer

class FraudPipeline:
    def __init__(self, random_state: int = 42):
        self.prep = DataPreprocessor()
        self.fe = FeatureEngineer()
        self.trainer = Trainer(random_state=random_state)

    def run(self, path: str) -> Dict[str, object]:
        # Step 1: Load & clean
        df = pd.read_csv(path)
        df = self.prep.clean_fraud_data(path)
        df = self.prep.enrich_with_country(df, "data/raw/IpAddress_to_Country.csv")

        # Step 2: Feature engineering
        df = self.fe.add_purchase_count(df)
        df = self.fe.add_time_features(df)

        # Step 3: Prepare training
        X, y = self.trainer.prepare_ecom(df)
        X_train_res, X_test_transformed, y_train_res, y_test = X

        # Step 4: Save arrays
        import numpy as np
        np.save("data/processed/X_train.npy", X_train_res)
        np.save("data/processed/X_test.npy", X_test_transformed)
        np.save("data/processed/y_train.npy", y_train_res)
        np.save("data/processed/y_test.npy", y_test)

        return {
            "X_train": X_train_res,
            "X_test": X_test_transformed,
            "y_train": y_train_res,
            "y_test": y_test
        }
