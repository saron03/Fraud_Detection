import pandas as pd

class FeatureEngineer:
    """Adds fraud-specific engineered features."""

    def add_purchase_count(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "user_id" in df.columns and "purchase_time" in df.columns:
            df["purchase_count"] = df.groupby("user_id")["purchase_time"].transform("count")
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)
        df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)

        df["hour_of_day"] = df["purchase_time"].dt.hour
        df["day_of_week"] = df["purchase_time"].dt.dayofweek
        df["time_since_signup"] = (df["purchase_time"] - df["signup_time"]).dt.total_seconds() / 3600.0
        return df

    def finalize(self, df: pd.DataFrame, save_path: str = None) -> pd.DataFrame:
        if save_path:
            df.to_csv(save_path, index=False)
        return df
