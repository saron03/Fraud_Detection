import pandas as pd
from src.fraud_detection.data_preprocessing import DataPreprocessor

def test_clean_fraud_data_removes_duplicates(tmp_path):
    df = pd.DataFrame({
        "user_id": [1,1],
        "signup_time": ["2020-01-01","2020-01-01"],
        "purchase_time": ["2020-01-01","2020-01-01"],
        "ip_address": [123,123],
        "class": [0,0]
    })
    file = tmp_path / "fraud.csv"
    df.to_csv(file, index=False)

    prep = DataPreprocessor()
    out = prep.clean_fraud_data(str(file))
    assert out.shape[0] == 1  # duplicates removed

def test_enrich_with_country(tmp_path):
    df = pd.DataFrame({"ip_address": [150]})
    ip_df = pd.DataFrame({
        "lower_bound_ip_address": [100],
        "upper_bound_ip_address": [200],
        "country": ["Testland"]
    })
    df_file = tmp_path / "fraud.csv"
    ip_file = tmp_path / "ip.csv"
    df.to_csv(df_file, index=False)
    ip_df.to_csv(ip_file, index=False)

    prep = DataPreprocessor()
    df_loaded = pd.read_csv(df_file)
    out = prep.enrich_with_country(df_loaded, str(ip_file))
    assert "country" in out.columns
    assert out.loc[0, "country"] == "Testland"
