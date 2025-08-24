import pandas as pd
from typing import Optional

class DataPreprocessor:
    """Handles cleaning, deduplication, and enrichment for fraud datasets."""

    def clean_fraud_data(self, fraud_path: str, save_path: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(fraud_path)
        df = df.drop_duplicates()

        # Fix date columns
        if "signup_time" in df.columns:
            df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)
        if "purchase_time" in df.columns:
            df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)

        if save_path:
            df.to_csv(save_path, index=False)
        return df

    def clean_creditcard_data(self, cc_path: str, save_path: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(cc_path)
        df = df.drop_duplicates()
        if save_path:
            df.to_csv(save_path, index=False)
        return df

    def enrich_with_country(
        self,
        fraud_df: pd.DataFrame,
        ip_country_path: str,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        df = fraud_df.copy()
        df["ip_int"] = df["ip_address"].astype(int)

        ip_country = pd.read_csv(ip_country_path).rename(
            columns={
                "lower_bound_ip_address": "lower",
                "upper_bound_ip_address": "upper"
            }
        )

        def find_country(ip: int):
            match = ip_country[(ip >= ip_country["lower"]) & (ip <= ip_country["upper"])]
            return match["country"].values[0] if not match.empty else "Unknown"

        df["country"] = df["ip_int"].apply(find_country)

        if save_path:
            df.to_csv(save_path, index=False)
        return df
