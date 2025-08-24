import pandas as pd
import numpy as np
from src.fraud_detection.model_training import Trainer

def test_prepare_ecom_and_smote():
    df = pd.DataFrame({
        "purchase_value":[100,200,150,300],
        "age":[25,30,22,28],
        "time_since_signup":[5,10,2,8],
        "hour_of_day":[10,15,9,12],
        "source":["web","web","mobile","mobile"],
        "browser":["chrome","safari","chrome","safari"],
        "sex":["M","F","M","F"],
        "country":["US","UK","US","UK"],
        "class":[0,0,1,1]
    })
    trainer = Trainer()
    X_train_res, X_test, y_train_res, y_test = trainer.prepare_ecom(df)
    assert X_train_res.shape[0] >= 4  # SMOTE increases rows
    assert len(y_train_res) == X_train_res.shape[0]

def test_train_logreg_and_eval():
    df = pd.DataFrame({
        "purchase_value":[100,200,150,300],
        "age":[25,30,22,28],
        "time_since_signup":[5,10,2,8],
        "hour_of_day":[10,15,9,12],
        "source":["web","web","mobile","mobile"],
        "browser":["chrome","safari","chrome","safari"],
        "sex":["M","F","M","F"],
        "country":["US","UK","US","UK"],
        "class":[0,0,1,1]
    })
    trainer = Trainer()
    X_train, X_test, y_train, y_test = trainer.prepare_ecom(df)
    model = trainer.train_logreg(X_train, y_train)
    results = trainer.evaluate(model, X_test, y_test, "Test")
    assert "f1" in results and "roc_auc" in results and "auc_pr" in results
