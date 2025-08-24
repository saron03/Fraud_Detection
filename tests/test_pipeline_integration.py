from src.fraud_detection.pipeline import FraudPipeline
import pandas as pd
import numpy as np

def test_full_pipeline(tmp_path):
    # Small synthetic dataset for integration test
    df = pd.DataFrame({
        "user_id":[1,2],
        "signup_time":["2020-01-01","2020-01-02"],
        "purchase_time":["2020-01-01","2020-01-02"],
        "ip_address":[123,456],
        "purchase_value":[100,200],
        "age":[25,30],
        "source":["web","mobile"],
        "browser":["chrome","safari"],
        "sex":["M","F"],
        "country":["US","UK"],
        "class":[0,1]
    })
    file = tmp_path / "data.csv"
    df.to_csv(file, index=False)

    # Run full pipeline
    pipe = FraudPipeline()
    outputs = pipe.run(str(file))

    # Assertions
    assert "X_train" in outputs
    assert "y_train" in outputs
    assert outputs["X_train"].shape[0] == outputs["y_train"].shape[0]
    assert outputs["X_test"].shape[0] == outputs["y_test"].shape[0]
