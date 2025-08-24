import pandas as pd
from src.fraud_detection.feature_engineering import FeatureEngineer

def test_add_purchase_count():
    df = pd.DataFrame({
        "user_id": [1,1,2],
        "purchase_time": ["2020-01-01","2020-01-02","2020-01-01"],
        "signup_time": ["2020-01-01","2020-01-01","2020-01-01"]
    })
    fe = FeatureEngineer()
    out = fe.add_purchase_count(df)
    assert "purchase_count" in out.columns
    assert out[out["user_id"]==1]["purchase_count"].iloc[0] == 2

def test_add_time_features():
    df = pd.DataFrame({
        "purchase_time": ["2020-01-01 11:00:00"],
        "signup_time": ["2020-01-01 10:00:00"]
    })
    fe = FeatureEngineer()
    out = fe.add_time_features(df)
    assert "hour_of_day" in out.columns
    assert "day_of_week" in out.columns
    assert "time_since_signup" in out.columns
    assert out["time_since_signup"].iloc[0] == 1.0
